import torch
import torch.nn as nn

from spiral_block import SpiralBlock
from position_encoding import SinusoidalPositionEncoding


class InformationSpiralTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        layers=6,
        max_sequence_length=2048,
        position_encoding="rope",
        use_memory_fusion=True,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = position_encoding
        self.position_embedding = (
            nn.Embedding(max_sequence_length, hidden_size)
            if position_encoding == "absolute" else None
        )
        self.sinusoidal_position = (
            SinusoidalPositionEncoding(hidden_size)
            if position_encoding == "sinusoidal" else None
        )
        self.blocks = nn.ModuleList(
            [SpiralBlock(hidden_size, position_encoding, use_memory_fusion) for _ in range(layers)]
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens):
        sequence_length = tokens.size(1)
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_sequence_length {self.max_sequence_length}"
            )
        x = self.embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=tokens.device)
            x = x + self.position_embedding(positions)[None, :, :]
        elif self.sinusoidal_position is not None:
            x = x + self.sinusoidal_position(
                sequence_length, x.device, x.dtype
            )[None, :, :]
        memory = None

        for block in self.blocks:
            x, memory = block(x, memory)

        return self.output(x)

    def memory_diversity_loss(self):
        return torch.stack(
            [block.memory.auxiliary_loss for block in self.blocks]
        ).mean()
