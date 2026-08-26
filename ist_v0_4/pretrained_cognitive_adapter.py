"""Frozen causal-LM adapter for IST v0.4 cognitive event memory."""
from __future__ import annotations

from contextlib import nullcontext
import torch
import torch.nn as nn

from cognitive_event_memory import CognitiveEventMemory
from config import CognitiveMemoryConfig


class FrozenCognitiveIST(nn.Module):
    def __init__(self, backbone, config: CognitiveMemoryConfig | None = None, injection_layer=-4):
        super().__init__()
        self.backbone = backbone
        self.config = config or CognitiveMemoryConfig()
        hidden = int(backbone.config.hidden_size)
        self.memory = CognitiveEventMemory(hidden, self.config)
        self.query_norm = nn.LayerNorm(hidden)
        parameter = next(backbone.parameters())
        self.memory.to(device=parameter.device, dtype=parameter.dtype)
        self.query_norm.to(device=parameter.device, dtype=parameter.dtype)
        self.injection_gate = nn.Parameter(torch.tensor(-0.01, dtype=torch.float32))
        layers = backbone.model.layers
        self.injection_layer = injection_layer if injection_layer >= 0 else len(layers) + injection_layer
        self._state = None
        self._intervention = "normal"
        self._captured = None
        self.last_provenance = None
        for item in backbone.parameters():
            item.requires_grad_(False)
        backbone.eval()

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def _hook(self, _module, args, kwargs):
        hidden = args[0]
        self._captured = hidden.detach()
        if self._state is None:
            return args, kwargs
        context, provenance = self.memory.read(self.query_norm(hidden), self._state, self._intervention)
        self.last_provenance = provenance
        gate = torch.tanh(self.injection_gate.float()).to(hidden.dtype)
        return (hidden + gate * context, *args[1:]), kwargs

    def forward(self, input_ids, state=None, chunk_id=0, position_offset=0,
                intervention="normal", detach_state=False):
        self._state, self._intervention = state, intervention
        self._captured = self.last_provenance = None
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
            raise RuntimeError("v0.4 source-layer hook did not execute")
        state = self.memory.write(self._captured, input_ids, state, chunk_id, position_offset)
        if detach_state:
            state = self.memory.detach_state(state)
        return logits, state

