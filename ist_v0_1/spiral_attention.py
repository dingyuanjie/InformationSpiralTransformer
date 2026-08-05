import torch
import torch.nn as nn

from rotary_attention import RotaryAttention


class SpiralAttention(nn.Module):
    """Attention over current information and historical memory."""

    def __init__(self, hidden_size, heads=8, position_encoding="rope"):
        super().__init__()
        self.position_encoding = position_encoding
        self.attention = (
            RotaryAttention(
                hidden_size, heads,
                rope_scale=4.0 if position_encoding == "scaled_rope" else 1.0,
                dynamic_base=64 if position_encoding == "dynamic_rope" else None,
            )
            if position_encoding in ("rope", "scaled_rope", "dynamic_rope")
            else nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        )

    def forward(self, x, memory):
        context = torch.cat([x, memory], dim=1)
        if self.position_encoding in ("rope", "scaled_rope", "dynamic_rope"):
            return self.attention(x, context, rotary_key_tokens=x.size(1))
        out, _ = self.attention(x, context, context, need_weights=False)
        return out
