import torch
import torch.nn as nn
import torch.nn.functional as F

from spiral_attention import SpiralAttention
from spiral_memory import SpiralMemory


class SpiralBlock(nn.Module):
    def __init__(self, hidden_size, position_encoding="rope", use_memory_fusion=True):
        super().__init__()
        self.attention = SpiralAttention(hidden_size, position_encoding=position_encoding)
        self.memory = SpiralMemory(hidden_size)
        self.use_memory_fusion = use_memory_fusion
        self.memory_read = (
            nn.MultiheadAttention(hidden_size, 8, batch_first=True)
            if use_memory_fusion else None
        )
        self.memory_fusion_gate = (
            nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
            if use_memory_fusion else None
        )
        self.capture_memory_read_weights = False
        self.last_memory_read_weights = None
        self.historical_read_scale = 1.0
        self.historical_consistency_threshold = None
        self.historical_consistency_temperature = 0.1
        self.last_historical_read_multiplier = None
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, memory=None):
        historical_memory = memory if memory is not None else x
        historical_multiplier = torch.as_tensor(
            self.historical_read_scale, device=x.device, dtype=x.dtype
        )
        if memory is not None and self.historical_consistency_threshold is not None:
            consistency = F.cosine_similarity(x.mean(dim=1), memory.mean(dim=1), dim=-1)
            adaptive = torch.sigmoid(
                (consistency - self.historical_consistency_threshold)
                / self.historical_consistency_temperature
            )
            historical_multiplier = historical_multiplier * adaptive[:, None, None]
        historical_memory = historical_memory * historical_multiplier
        self.last_historical_read_multiplier = historical_multiplier.float().mean().detach()
        attn = self.attention(x, historical_memory)
        x = self.norm1(x + attn)

        new_memory, memory_feature = self.memory(x, memory)
        if self.use_memory_fusion:
            if self.capture_memory_read_weights:
                memory_context, read_weights = self.memory_read(
                    x, new_memory, new_memory, need_weights=True,
                    average_attn_weights=False,
                )
                self.last_memory_read_weights = read_weights.detach()
            else:
                memory_context, _ = self.memory_read(
                    x, new_memory, new_memory, need_weights=False
                )
                self.last_memory_read_weights = None
            fusion_gate = self.memory_fusion_gate(
                torch.cat([x, memory_context], dim=-1)
            )
            memory_feature = memory_feature + fusion_gate * memory_context
            self.last_fusion_gate = fusion_gate.detach()
        x = self.norm2(x + self.ffn(memory_feature))
        return x, new_memory
