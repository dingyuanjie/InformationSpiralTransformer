import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions, scale=1.0):
    head_dim = x.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    frequencies = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device) / head_dim)
    )
    angles = (positions[:, None] / scale) * frequencies[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class RotaryAttention(nn.Module):
    def __init__(self, hidden_size, heads=8, rope_scale=1.0, dynamic_base=None):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.rope_scale = rope_scale
        self.dynamic_base = dynamic_base
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _split(self, x):
        return x.view(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, context, rotary_key_tokens=None):
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(context))
        v = self._split(self.v_proj(context))
        scale = self.rope_scale
        if self.dynamic_base is not None:
            scale = max(1.0, query.size(1) / self.dynamic_base)
        q = apply_rope(q, torch.arange(query.size(1), device=query.device), scale)
        key_tokens = context.size(1) if rotary_key_tokens is None else rotary_key_tokens
        rotated = apply_rope(
            k[:, :, :key_tokens],
            torch.arange(key_tokens, device=query.device),
            scale,
        )
        k = torch.cat((rotated, k[:, :, key_tokens:]), dim=2)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(query.size(0), query.size(1), -1)
        return self.out_proj(output)
