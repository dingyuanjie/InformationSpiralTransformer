import torch
import torch.nn as nn

from spiral_attention import SpiralAttention
from spiral_memory import SpiralMemory


class SpiralBlock(nn.Module):
    def __init__(self, hidden_size, position_encoding="rope", use_memory_fusion=True):
        super().__init__()
        self.attention = SpiralAttention(hidden_size, position_encoding=position_encoding)
        self.memory = SpiralMemory(hidden_size)
        self.use_memory_fusion = use_memory_fusion
        self.memory_read = nn.MultiheadAttention(
            hidden_size, 8, batch_first=True
        )
        self.memory_fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, memory=None):
        attn = self.attention(x, memory if memory is not None else x)
        x = self.norm1(x + attn)

        new_memory, memory_feature = self.memory(x, memory)
        if self.use_memory_fusion:
            memory_context, _ = self.memory_read(
                x, new_memory, new_memory, need_weights=False
            )
            fusion_gate = self.memory_fusion_gate(
                torch.cat([x, memory_context], dim=-1)
            )
            memory_feature = memory_feature + fusion_gate * memory_context
            self.last_fusion_gate = fusion_gate.detach()
        x = self.norm2(x + self.ffn(memory_feature))
        return x, new_memory
