"""Frozen Qwen with causal Fast-Memory injection inside an upper decoder block."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from hierarchical_memory import HierarchicalMemory
from run_pretrained_base_smoke_0_3_1 import fast_only_config


class FrozenLayerInjectedIST(nn.Module):
    """Keep Qwen frozen and inject historical Fast slots before one decoder layer.

    The first chunk has no historical state and therefore follows the exact frozen
    backbone path.  Its final hidden states write Fast Memory.  On later chunks a
    trainable cross-attention bridge reads only the *incoming* Fast state and
    injects the result before ``injection_layer``.  Memory written by the current
    chunk cannot affect that same chunk's injection.
    """

    def __init__(self, backbone: nn.Module, injection_layer: int = -4, heads: int = 8,
                 layer_matched_write: bool = False):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.config.hidden_size)
        self.memory = HierarchicalMemory(self.hidden_size, fast_only_config())
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.layer_read = nn.MultiheadAttention(
            self.hidden_size, heads, batch_first=True, bias=False
        )
        self.layer_out = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.injection_scale = nn.Parameter(torch.tensor(-0.01, dtype=torch.float32))
        self.layer_matched_write = layer_matched_write
        self.intervention = "normal"
        self.hook_calls = 0
        self.last_injection_norm = 0.0
        self.last_pre_injection = None
        self.last_post_injection = None
        self.last_layer_input_sequence = None

        layers = self.backbone.model.layers
        resolved = injection_layer if injection_layer >= 0 else len(layers) + injection_layer
        if not 0 <= resolved < len(layers):
            raise ValueError(f"injection layer {injection_layer} resolves outside {len(layers)} layers")
        self.injection_layer = resolved
        self.total_layers = len(layers)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        modules = (self.memory, self.query_norm, self.memory_norm, self.layer_read, self.layer_out)
        return [parameter for module in modules for parameter in module.parameters()] + [self.injection_scale]

    def trainable_state_dict(self):
        return {
            "memory": self.memory.state_dict(),
            "query_norm": self.query_norm.state_dict(),
            "memory_norm": self.memory_norm.state_dict(),
            "layer_read": self.layer_read.state_dict(),
            "layer_out": self.layer_out.state_dict(),
            "injection_scale": self.injection_scale.detach(),
        }

    def load_trainable_state_dict(self, state):
        for name in ("memory", "query_norm", "memory_norm", "layer_read", "layer_out"):
            getattr(self, name).load_state_dict(state[name])
        self.injection_scale.data.copy_(state["injection_scale"].float())

    def _historical_fast(self, state):
        if state is None:
            return None
        memory = state["fast"]
        if self.intervention in {"zero_fast", "zero_memory"}:
            # Bypass the bridge entirely.  Passing zero slots through trainable
            # normalization could otherwise create a condition-independent bias.
            return None
        elif self.intervention == "roll_fast":
            memory = torch.roll(memory, 1, dims=1)
        elif self.intervention == "swap_fast" and memory.size(0) > 1:
            memory = torch.roll(memory, 1, dims=0)
        return memory

    def _pre_hook(self, historical_fast):
        def inject(_module, args, kwargs):
            hidden = args[0]
            # Capture the exact representation space at the injection boundary.
            # The detached sequence is also the write source in Level 0.5.3.
            self.last_layer_input_sequence = hidden.detach()
            if historical_fast is None:
                return args, kwargs
            memory = historical_fast.to(hidden.dtype)
            context, _ = self.layer_read(
                self.query_norm(hidden), self.memory_norm(memory), self.memory_norm(memory),
                need_weights=False,
            )
            delta = self.layer_out(context)
            scale = torch.tanh(self.injection_scale.float()).to(hidden.dtype)
            injected = hidden + scale * delta
            self.last_pre_injection = hidden[:, -1]
            self.last_post_injection = injected[:, -1]
            self.hook_calls += 1
            self.last_injection_norm = float((scale * delta).detach().float().norm(dim=-1).mean())
            return (injected, *args[1:]), kwargs

        return inject

    def forward(self, input_ids, state=None, intervention="normal", detach_state=False):
        self.intervention = intervention
        historical_fast = self._historical_fast(state)
        use_injection = historical_fast is not None and intervention != "reset_memory"
        self.last_injection_norm = 0.0
        self.last_pre_injection = None
        self.last_post_injection = None
        self.last_layer_input_sequence = None
        layer = self.backbone.model.layers[self.injection_layer]
        # Always hook the layer: without history this is a capture-only hook and
        # returns the input unchanged, preserving the exact frozen-Base path.
        handle = layer.register_forward_pre_hook(
            self._pre_hook(historical_fast if use_injection else None), with_kwargs=True
        )
        try:
            context = nullcontext() if use_injection else torch.no_grad()
            with context:
                output = self.backbone.model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                )
                hidden = output.last_hidden_state
                logits = self.backbone.get_output_embeddings()(hidden[:, -1:])
        finally:
            handle.remove()

        # The write happens after the read/injection, enforcing chunk causality.
        if self.layer_matched_write:
            if self.last_layer_input_sequence is None:
                raise RuntimeError("injection-layer write source was not captured")
            write_hidden = self.last_layer_input_sequence
        else:
            write_hidden = hidden.detach()
        state, _ = self.memory(write_hidden, state)
        if detach_state:
            state = self.memory.detach_state(state)
        return logits, state

    def clear_intervention(self):
        self.intervention = "normal"
        self.memory.intervention = "normal"
