"""IST v0.2 hierarchical internal-Memory model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HierarchicalMemoryConfig
from hierarchical_block import HierarchicalSpiralBlock
from position_encoding import SinusoidalPositionEncoding


class HierarchicalInformationSpiralTransformer(nn.Module):
    memory_arch = "hierarchical_v0_2"

    def __init__(self, vocab_size, hidden_size=512, layers=6, max_sequence_length=2048,
                 position_encoding="rope", hierarchical_config=None):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.hierarchical_config = (hierarchical_config if isinstance(hierarchical_config, HierarchicalMemoryConfig)
                                    else HierarchicalMemoryConfig.from_dict(hierarchical_config))
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = position_encoding
        self.position_embedding = (nn.Embedding(max_sequence_length, hidden_size)
                                   if position_encoding == "absolute" else None)
        self.sinusoidal_position = (SinusoidalPositionEncoding(hidden_size)
                                    if position_encoding == "sinusoidal" else None)
        self.blocks = nn.ModuleList([
            HierarchicalSpiralBlock(hidden_size, self.hierarchical_config, position_encoding)
            for _ in range(layers)])
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, memory=None, return_memory=False, detach_memory=False,
                per_layer_memory=True, return_diagnostics=False):
        length = tokens.size(1)
        if length > self.max_sequence_length:
            raise ValueError(f"sequence length {length} exceeds max_sequence_length {self.max_sequence_length}")
        hidden = self.embedding(tokens)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(torch.arange(length, device=tokens.device))[None]
        elif self.sinusoidal_position is not None:
            hidden = hidden + self.sinusoidal_position(length, hidden.device, hidden.dtype)[None]
        states = [None] * len(self.blocks) if memory is None else list(memory)
        if len(states) != len(self.blocks):
            raise ValueError("hierarchical Memory requires one state dictionary per layer")
        new_states = []
        for block, state in zip(self.blocks, states):
            hidden, state = block(hidden, state)
            new_states.append(block.memory.detach_state(state) if detach_memory else state)
        logits = self.output(hidden)
        result = (logits, new_states) if return_memory else logits
        if return_diagnostics:
            diagnostics = [block.memory.last_diagnostics for block in self.blocks]
            return (*result, diagnostics) if isinstance(result, tuple) else (result, diagnostics)
        return result

    def memory_diversity_loss(self):
        losses = []
        for block in self.blocks:
            fast = block.memory.fast_writer.auxiliary_loss
            losses.append(fast)
        return torch.stack(losses).mean()

    def set_memory_intervention(self, intervention: str, layer: int | None = None):
        targets = self.blocks if layer is None else (self.blocks[layer],)
        for block in targets:
            block.memory.intervention = intervention

    def clear_memory_interventions(self):
        self.set_memory_intervention("normal")


def transfer_v0_1_weights(hierarchical, legacy_state: dict):
    """Transfer only shape-compatible shared and Fast-Memory weights."""
    current = hierarchical.state_dict()
    transferred = {}
    for key, value in legacy_state.items():
        candidate = key
        candidate = candidate.replace(".memory.encoder.", ".memory.fast_writer.encoder.")
        candidate = candidate.replace(".memory.slot_queries", ".memory.fast_writer.slot_queries")
        candidate = candidate.replace(".memory.memory_key.", ".memory.fast_writer.memory_key.")
        candidate = candidate.replace(".memory.memory_attention.", ".memory.fast_writer.memory_attention.")
        candidate = candidate.replace(".memory.update_gate.", ".memory.fast_writer.update_gate.")
        candidate = candidate.replace(".memory_read.", ".memory.fast_read.")
        candidate = candidate.replace(".memory_fusion_gate.", ".memory.fast_fusion.")
        if candidate in current and current[candidate].shape == value.shape:
            transferred[candidate] = value
    missing, unexpected = hierarchical.load_state_dict(transferred, strict=False)
    return {"transferred": sorted(transferred), "missing": missing, "unexpected": unexpected}
