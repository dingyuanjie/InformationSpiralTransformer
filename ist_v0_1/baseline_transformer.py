import torch
import torch.nn as nn

from rotary_attention import RotaryAttention
from position_encoding import SinusoidalPositionEncoding


class StandardBlock(nn.Module):
    def __init__(self, hidden_size, heads, dropout, position_encoding):
        super().__init__()
        self.position_encoding = position_encoding
        self.attention = (
            RotaryAttention(
                hidden_size, heads,
                rope_scale=4.0 if position_encoding == "scaled_rope" else 1.0,
                dynamic_base=64 if position_encoding == "dynamic_rope" else None,
            )
            if position_encoding in ("rope", "scaled_rope", "dynamic_rope")
            else nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        if self.position_encoding in ("rope", "scaled_rope", "dynamic_rope"):
            attended = self.attention(x, x)
        else:
            attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class StandardTransformer(nn.Module):
    """A parameter-comparable encoder-only Transformer baseline."""

    def __init__(
        self,
        vocab_size,
        hidden_size=64,
        layers=2,
        heads=8,
        max_sequence_length=2048,
        dropout=0.0,
        position_encoding="rope",
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.max_sequence_length = max_sequence_length
        self.position_embedding = (
            nn.Embedding(max_sequence_length, hidden_size)
            if position_encoding == "absolute" else None
        )
        self.sinusoidal_position = (
            SinusoidalPositionEncoding(hidden_size)
            if position_encoding == "sinusoidal" else None
        )
        self.blocks = nn.ModuleList([
            StandardBlock(hidden_size, heads, dropout, position_encoding)
            for _ in range(layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        sequence_length = tokens.size(1)
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_sequence_length {self.max_sequence_length}"
            )
        hidden = self.token_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=tokens.device)
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        elif self.sinusoidal_position is not None:
            hidden = hidden + self.sinusoidal_position(
                sequence_length, hidden.device, hidden.dtype
            )[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)
