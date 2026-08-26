"""Frozen causal-LM adapter using direct source-token Memory at one decoder layer."""
from __future__ import annotations

from contextlib import nullcontext
import torch
import torch.nn as nn

from config import SourceTokenMemoryConfig
from source_token_memory import SourceTokenMemory


class FrozenTokenMemoryIST(nn.Module):
    def __init__(self, backbone, config: SourceTokenMemoryConfig | None = None):
        super().__init__()
        self.backbone = backbone
        self.config = config or SourceTokenMemoryConfig()
        hidden = int(backbone.config.hidden_size)
        self.memory = SourceTokenMemory(hidden, self.config)
        self.query_norm = nn.LayerNorm(hidden)
        backbone_parameter = next(backbone.parameters())
        # ``Module.to(device)`` does not align the dtype of newly constructed
        # adapter layers with a BF16/FP16 backbone. Do it here so the first
        # cross-chunk read cannot mix Float32 LayerNorm weights with BF16 states.
        self.memory.to(device=backbone_parameter.device, dtype=backbone_parameter.dtype)
        self.query_norm.to(device=backbone_parameter.device, dtype=backbone_parameter.dtype)
        self.injection_gate = nn.Parameter(torch.tensor(self.config.initial_gate, dtype=torch.float32))
        layers = backbone.model.layers
        layer = self.config.injection_layer
        self.injection_layer = layer if layer >= 0 else len(layers) + layer
        self._state = None
        self._intervention = "normal"
        self._captured = None
        self.last_provenance = None
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _hook(self, _module, args, kwargs):
        hidden = args[0]
        self._captured = hidden.detach()
        if self._state is None:
            return args, kwargs
        context, provenance = self.memory.read(
            self.query_norm(hidden), self._state, self._intervention
        )
        self.last_provenance = provenance
        gate = torch.tanh(self.injection_gate.float()).to(hidden.dtype)
        return (hidden + gate * context, *args[1:]), kwargs

    def forward(self, input_ids, state=None, chunk_id=0, position_offset=0,
                intervention="normal", detach_state=False):
        self._state = state
        self._intervention = intervention
        self._captured = None
        self.last_provenance = None
        handle = self.backbone.model.layers[self.injection_layer].register_forward_pre_hook(
            self._hook, with_kwargs=True
        )
        try:
            context = nullcontext() if state is not None else torch.no_grad()
            with context:
                output = self.backbone.model(input_ids=input_ids, use_cache=False, return_dict=True)
                hidden = output.last_hidden_state
                logits = self.backbone.get_output_embeddings()(hidden[:, -1:])
        finally:
            handle.remove()
        if self._captured is None:
            raise RuntimeError("source-layer hook did not execute")
        state = self.memory.write(
            self._captured, input_ids, state, chunk_id=chunk_id,
            position_offset=position_offset,
        )
        if detach_state:
            state = self.memory.detach_state(state)
        return logits, state
