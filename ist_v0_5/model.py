"""Small Level A model used to validate the v0.5 memory algorithm."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import V05Config
from hybrid_memory import HybridEvidenceCoreMemory, detach_state
from strict_data import FACT


class HybridIST(nn.Module):
    def __init__(self, config: V05Config, variant: str = "hybrid"):
        super().__init__()
        if variant not in {"hybrid", "evidence_only", "core_only", "no_memory", "last_k"}:
            raise ValueError(f"unknown variant {variant}")
        self.config, self.variant = config, variant
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        layer = nn.TransformerEncoderLayer(config.hidden_size, config.heads,
                                           config.hidden_size * 4, config.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.memory = HybridEvidenceCoreMemory(config)
        self.query_norm = nn.LayerNorm(config.hidden_size)
        self.fusion = nn.Sequential(nn.Linear(config.hidden_size * 2, config.hidden_size), nn.GELU(),
                                    nn.LayerNorm(config.hidden_size))
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.last_provenance = None

    def encode(self, tokens):
        return self.encoder(self.embedding(tokens))

    def build_state(self, history, detach_between_chunks=False, intervention="normal",
                    oracle_positions=None):
        # history: [batch, chunks, tokens]. A later query is deliberately absent.
        state = None
        for chunk in range(history.size(1)):
            tokens = history[:, chunk]
            hidden = self.encode(tokens)
            windows = tokens.unfold(1, self.config.evidence_span, 1)
            candidate_mask = windows[:, :, 0].eq(FACT)
            state = self.memory.write(hidden, tokens, state, chunk,
                                      chunk * history.size(2),
                                      block_writer=intervention == "block_writer",
                                      candidate_mask=candidate_mask)
            if detach_between_chunks:
                state = detach_state(state)
        if oracle_positions is not None:
            state = self.force_oracle_evidence(history, state, oracle_positions)
        return state

    def force_oracle_evidence(self, history, state, oracle_positions):
        """Diagnostic only: force the supervised target occurrence into one Evidence slot."""
        batch = history.size(0); width = history.size(2); device = history.device
        chunks = torch.div(oracle_positions, width, rounding_mode="floor")
        starts = oracle_positions.remainder(width)
        source_tokens = history[torch.arange(batch, device=device), chunks]
        source_hidden = self.encode(source_tokens)
        offsets = starts[:, None] + torch.arange(self.config.evidence_span, device=device)[None]
        target_values = source_hidden.gather(1, offsets[..., None].expand(-1, -1, self.config.hidden_size))
        target_ids = source_tokens.gather(1, offsets)
        evidence = {key: value.clone() for key, value in state["evidence"].items()}
        slot = self.config.evidence_capacity - 1
        evidence["values"][:, slot] = target_values
        evidence["token_ids"][:, slot] = target_ids
        evidence["positions"][:, slot] = offsets + chunks[:, None] * width
        evidence["source_chunks"][:, slot] = chunks
        evidence["born"][:, slot] = chunks + 1
        evidence["last_read"][:, slot] = int(state["clock"])
        evidence["usage"][:, slot] = 0
        evidence["importance"][:, slot] = 1e6
        evidence["valid"][:, slot] = True
        return {"evidence": evidence, "core": state["core"], "clock": state["clock"]}

    def forward(self, history, query, state=None, intervention="normal",
                source_chunk=None, detach_between_chunks=False, oracle_positions=None):
        if state is None:
            state = self.build_state(history, detach_between_chunks, intervention, oracle_positions)
        query_hidden = self.query_norm(self.encode(query))
        query_summary = query_hidden.mean(1)
        if self.variant == "no_memory":
            context, provenance = torch.zeros_like(query_summary), None
        elif self.variant == "last_k":
            flat = history.reshape(history.size(0), -1)
            tail = flat[:, -self.config.evidence_span * self.config.reads_per_query:]
            context, provenance = self.embedding(tail).mean(1), None
        else:
            effective = intervention
            if self.variant == "evidence_only" and intervention == "normal": effective = "zero_core"
            if self.variant == "core_only" and intervention == "normal": effective = "zero_evidence"
            context, provenance = self.memory.read(query_hidden, state, effective, source_chunk)
        self.last_provenance = provenance
        fused = self.fusion(torch.cat((query_summary, context), dim=-1))
        return self.output(fused), state
