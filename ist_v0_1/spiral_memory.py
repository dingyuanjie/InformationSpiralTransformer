import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpiralMemory(nn.Module):
    """Information Spiral Memory v0.2 with specialized memory slots."""

    def __init__(self, hidden_size, memory_slots=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.slot_queries = nn.Parameter(
            torch.randn(memory_slots, hidden_size) / math.sqrt(hidden_size)
        )
        self.memory_key = nn.Linear(hidden_size, hidden_size, bias=False)

        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )

        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        # Populated after forward for diagnostics and visualization only.
        self.last_diagnostics = None
        self.capture_memory_attention_weights = False

    def initialize_memory(self, batch_size, device):
        return torch.zeros(
            batch_size,
            self.memory_slots,
            self.hidden_size,
            device=device,
        )

    def forward(self, hidden, memory=None):
        """
        Args:
            hidden: Tensor shaped [batch, seq, hidden].
            memory: Optional tensor shaped [batch, memory_slots, hidden].

        Returns:
            A tuple of (new_memory, fused_hidden).
        """
        batch = hidden.size(0)

        if memory is None:
            memory = self.initialize_memory(batch, hidden.device)

        encoded = self.encoder(hidden)
        if self.capture_memory_attention_weights:
            attended_memory, memory_attention_weights = self.memory_attention(
                encoded,
                memory,
                memory,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attended_memory, _ = self.memory_attention(
                encoded,
                memory,
                memory,
                need_weights=False,
            )
            memory_attention_weights = None
        fused = encoded + attended_memory

        keys = self.memory_key(fused)
        score = torch.einsum(
            "bth,sh->bst", keys, self.slot_queries
        ) / math.sqrt(self.hidden_size)
        compression_weights = torch.softmax(score, dim=-1)
        compressed = compression_weights @ fused

        # Each slot chooses independently what to retain and replace.
        gate = self.update_gate(
            torch.cat([memory, compressed], dim=-1)
        )

        new_memory = gate * compressed + (1 - gate) * memory
        normalized_memory = F.normalize(new_memory, dim=-1)
        gram = normalized_memory @ normalized_memory.transpose(-1, -2)
        identity = torch.eye(
            self.memory_slots, device=gram.device, dtype=gram.dtype
        ).unsqueeze(0)
        diversity_loss = ((gram - identity) ** 2).sum(dim=(-1, -2))
        diversity_loss = diversity_loss.mean() / (
            self.memory_slots * max(self.memory_slots - 1, 1)
        )
        safe_weights = compression_weights.clamp_min(1e-9)
        attention_entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)
        self.last_diagnostics = {
            "compression_weights": compression_weights.detach(),
            "update_gate": gate.detach(),
            "old_memory": memory.detach(),
            "new_memory": new_memory.detach(),
            "attention_entropy": attention_entropy.detach(),
            "diversity_loss": diversity_loss.detach(),
            "memory_attention_weights": (
                memory_attention_weights.detach()
                if memory_attention_weights is not None else None
            ),
        }
        self.auxiliary_loss = diversity_loss
        return new_memory, fused
