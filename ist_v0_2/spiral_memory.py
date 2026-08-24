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
        # Inference-time controls used by causal robustness experiments.
        self.propagation_scale = 1.0
        self.propagation_relative_cap = None
        self.propagation_consistency_threshold = None
        self.propagation_consistency_temperature = 0.1
        # Disabled-by-default control for slot-geometry causal experiments.
        self.slot_decorrelation_strength = 0.0

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
        propagation_multiplier = torch.as_tensor(
            self.propagation_scale, device=encoded.device, dtype=encoded.dtype
        )
        if self.propagation_consistency_threshold is not None:
            consistency = F.cosine_similarity(encoded, attended_memory, dim=-1)
            adaptive_multiplier = torch.sigmoid(
                (consistency - self.propagation_consistency_threshold)
                / self.propagation_consistency_temperature
            )
            propagation_multiplier = propagation_multiplier * adaptive_multiplier.unsqueeze(-1)
        if self.propagation_relative_cap is not None:
            encoded_norm = encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            attended_norm = attended_memory.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            relative_multiplier = self.propagation_relative_cap * encoded_norm / attended_norm
            propagation_multiplier = propagation_multiplier * relative_multiplier.clamp(max=1.0)
        fused = encoded + propagation_multiplier * attended_memory

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
        if self.slot_decorrelation_strength:
            strength = float(self.slot_decorrelation_strength)
            if not 0.0 <= strength <= 1.0:
                raise ValueError("slot_decorrelation_strength must be in [0, 1]")
            source = new_memory.float()
            norms = source.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            orthogonal, _ = torch.linalg.qr(source.transpose(-1, -2), mode="reduced")
            orthogonal = orthogonal.transpose(-1, -2)
            signs = torch.sign((orthogonal * source).sum(dim=-1, keepdim=True))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            decorrelated = orthogonal * signs * norms
            new_memory = ((1.0 - strength) * source + strength * decorrelated).to(new_memory.dtype)
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
            "propagation_multiplier_mean": propagation_multiplier.float().mean().detach(),
            "encoded_norm": encoded.float().norm(dim=-1).mean(dim=-1).detach(),
            "attended_memory_norm": attended_memory.float().norm(dim=-1).mean(dim=-1).detach(),
            "propagation_ratio": (
                (propagation_multiplier.float() * attended_memory.float()).norm(dim=-1)
                / encoded.float().norm(dim=-1).clamp_min(1e-8)
            ).mean(dim=-1).detach(),
        }
        self.auxiliary_loss = diversity_loss
        return new_memory, fused
