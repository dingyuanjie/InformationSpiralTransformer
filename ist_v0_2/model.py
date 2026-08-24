import torch
import torch.nn as nn

from spiral_block import SpiralBlock
from position_encoding import SinusoidalPositionEncoding


class InformationSpiralTransformer(nn.Module):
    memory_arch = "v0_1"
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

    def forward(
        self,
        tokens,
        memory=None,
        return_memory=False,
        detach_memory=False,
        per_layer_memory=False,
    ):
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
        if per_layer_memory or isinstance(memory, (list, tuple)):
            memories = [None] * len(self.blocks) if memory is None else list(memory)
            if len(memories) != len(self.blocks):
                raise ValueError("per-layer memory count must equal model layer count")
            new_memories = []
            for index, block in enumerate(self.blocks):
                x, layer_memory = block(x, memories[index])
                new_memories.append(layer_memory)
            memory = new_memories
        else:
            for block in self.blocks:
                x, memory = block(x, memory)
        logits = self.output(x)
        if return_memory:
            if detach_memory:
                if isinstance(memory, list):
                    memory = [item.detach() for item in memory]
                else:
                    memory = memory.detach()
            return logits, memory
        return logits

    def memory_diversity_loss(self):
        return torch.stack(
            [block.memory.auxiliary_loss for block in self.blocks]
        ).mean()


def build_model(memory_arch="v0_1", **kwargs):
    """Compatibility factory; v0.1 returns the exact legacy state-dict layout."""
    if memory_arch == "v0_1":
        kwargs.pop("hierarchical_config", None)
        return InformationSpiralTransformer(**kwargs)
    if memory_arch == "hierarchical_v0_2":
        from hierarchical_model import HierarchicalInformationSpiralTransformer
        return HierarchicalInformationSpiralTransformer(**kwargs)
    raise ValueError("memory_arch must be v0_1 or hierarchical_v0_2")
