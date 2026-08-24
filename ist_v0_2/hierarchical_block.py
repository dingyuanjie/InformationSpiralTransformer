"""Transformer block backed by one hierarchical Memory lifecycle layer."""
from __future__ import annotations

import torch
import torch.nn as nn

from hierarchical_memory import HierarchicalMemory
from spiral_attention import SpiralAttention


class HierarchicalSpiralBlock(nn.Module):
    def __init__(self, hidden_size, config, position_encoding="rope"):
        super().__init__()
        self.attention = SpiralAttention(hidden_size, position_encoding=position_encoding)
        self.memory = HierarchicalMemory(hidden_size, config)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                                 nn.Linear(hidden_size * 4, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden, state=None):
        historical = hidden if state is None else state["fast"]
        attended = self.attention(hidden, historical)
        hidden = self.norm1(hidden + attended)
        new_state, memory_feature = self.memory(hidden, state)
        hidden = self.norm2(hidden + self.ffn(memory_feature))
        return hidden, new_state
