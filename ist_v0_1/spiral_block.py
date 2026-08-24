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
        # Disabled-by-default inference controls for causal Memory experiments.
        self.memory_read_topk = None
        self.memory_read_keep_slots = None
        self.memory_read_ablate_slots = None
        self.memory_read_slot_scales = None
        self.fusion_gate_floor = None
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
            read_memory = new_memory
            if self.memory_read_slot_scales:
                slot_scale = torch.ones(
                    new_memory.size(1), device=new_memory.device, dtype=new_memory.dtype
                )
                for slot, scale in self.memory_read_slot_scales.items():
                    slot_scale[int(slot)] = float(scale)
                read_memory = new_memory * slot_scale[None, :, None]
            explicit_mask = None
            if self.memory_read_keep_slots is not None or self.memory_read_ablate_slots is not None:
                explicit_mask = torch.zeros(
                    new_memory.size(0), new_memory.size(1), device=x.device, dtype=torch.bool
                )
                if self.memory_read_keep_slots is not None:
                    explicit_mask.fill_(True)
                    explicit_mask[:, list(self.memory_read_keep_slots)] = False
                if self.memory_read_ablate_slots is not None:
                    explicit_mask[:, list(self.memory_read_ablate_slots)] = True
                if explicit_mask.all(dim=1).any():
                    raise ValueError("explicit Memory slot routing cannot mask every slot")
            if explicit_mask is not None:
                memory_context, read_weights = self.memory_read(
                    x, read_memory, read_memory,
                    key_padding_mask=explicit_mask,
                    need_weights=self.capture_memory_read_weights,
                    average_attn_weights=False,
                )
                self.last_memory_read_weights = (
                    read_weights.detach() if read_weights is not None else None
                )
            elif self.memory_read_topk is not None:
                if not 1 <= int(self.memory_read_topk) <= new_memory.size(1):
                    raise ValueError("memory_read_topk must be between 1 and the number of slots")
                _, screening_weights = self.memory_read(
                    x, read_memory, read_memory, need_weights=True,
                    average_attn_weights=False,
                )
                slot_salience = screening_weights.float().mean(dim=(1, 2))
                selected = slot_salience.topk(int(self.memory_read_topk), dim=-1).indices
                key_padding_mask = torch.ones(
                    new_memory.size(0), new_memory.size(1), device=x.device, dtype=torch.bool
                )
                key_padding_mask.scatter_(1, selected, False)
                memory_context, read_weights = self.memory_read(
                    x, read_memory, read_memory,
                    key_padding_mask=key_padding_mask,
                    need_weights=self.capture_memory_read_weights,
                    average_attn_weights=False,
                )
                self.last_memory_read_weights = (
                    read_weights.detach() if read_weights is not None else None
                )
            elif self.capture_memory_read_weights:
                memory_context, read_weights = self.memory_read(
                    x, read_memory, read_memory, need_weights=True,
                    average_attn_weights=False,
                )
                self.last_memory_read_weights = read_weights.detach()
            else:
                memory_context, _ = self.memory_read(
                    x, read_memory, read_memory, need_weights=False
                )
                self.last_memory_read_weights = None
            fusion_gate = self.memory_fusion_gate(
                torch.cat([x, memory_context], dim=-1)
            )
            if self.fusion_gate_floor is not None:
                fusion_gate = fusion_gate.clamp_min(float(self.fusion_gate_floor))
            memory_feature = memory_feature + fusion_gate * memory_context
            self.last_fusion_gate = fusion_gate.detach()
        x = self.norm2(x + self.ffn(memory_feature))
        return x, new_memory
