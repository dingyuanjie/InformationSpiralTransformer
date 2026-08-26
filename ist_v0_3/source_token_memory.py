"""Direct, provenance-preserving source-token memory for IST v0.3."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SourceTokenMemoryConfig


class SourceTokenMemory(nn.Module):
    """Select source states without compressing them into learned slot vectors.

    Every retained value keeps its source token id, absolute stream position and
    chunk id. Selection is discrete and inspectable; retrieval is query-dependent.
    """

    def __init__(self, hidden_size: int, config: SourceTokenMemoryConfig | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config or SourceTokenMemoryConfig()
        self.config.validate(hidden_size)
        self.salience = nn.Linear(hidden_size, 1, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_diagnostics = None

    def empty_state(self, batch: int, device, dtype):
        capacity = self.config.capacity
        return {
            "values": torch.zeros(batch, capacity, self.hidden_size, device=device, dtype=dtype),
            "scores": torch.full((batch, capacity), -torch.inf, device=device),
            "token_ids": torch.full((batch, capacity), -1, device=device, dtype=torch.long),
            "positions": torch.full((batch, capacity), -1, device=device, dtype=torch.long),
            "chunk_ids": torch.full((batch, capacity), -1, device=device, dtype=torch.long),
            "valid": torch.zeros(batch, capacity, device=device, dtype=torch.bool),
            "chunks_written": 0,
        }

    @staticmethod
    def detach_state(state):
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in state.items()}

    def write(self, hidden, token_ids, state=None, chunk_id=0, position_offset=0):
        batch, tokens, _ = hidden.shape
        if state is None:
            state = self.empty_state(batch, hidden.device, hidden.dtype)
        # Remove the chunk-wide common direction only for scoring. Stored values
        # remain exact source-layer states and can always be traced to input tokens.
        residual = hidden.float() - hidden.float().mean(1, keepdim=True)
        learned = self.salience(residual.to(self.salience.weight.dtype)).squeeze(-1).float()
        novelty = residual.norm(dim=-1)
        scores = learned + novelty / novelty.mean(1, keepdim=True).clamp_min(1e-6)
        count = min(self.config.writes_per_chunk, tokens)
        selected_scores, selected = scores.topk(count, dim=1)
        selected_values = hidden.gather(1, selected[..., None].expand(-1, -1, self.hidden_size))
        selected_ids = token_ids.gather(1, selected)
        selected_positions = selected + int(position_offset)
        selected_chunks = torch.full_like(selected, int(chunk_id))

        values = torch.cat((state["values"], selected_values), dim=1)
        merged_scores = torch.cat((state["scores"], selected_scores), dim=1)
        ids = torch.cat((state["token_ids"], selected_ids), dim=1)
        positions = torch.cat((state["positions"], selected_positions), dim=1)
        chunks = torch.cat((state["chunk_ids"], selected_chunks), dim=1)
        valid = torch.cat((state["valid"], torch.ones_like(selected, dtype=torch.bool)), dim=1)
        merged_scores = merged_scores.masked_fill(~valid, -torch.inf)
        keep_scores, keep = merged_scores.topk(self.config.capacity, dim=1)
        gather_hidden = keep[..., None].expand(-1, -1, self.hidden_size)
        new_state = {
            "values": values.gather(1, gather_hidden),
            "scores": keep_scores,
            "token_ids": ids.gather(1, keep),
            "positions": positions.gather(1, keep),
            "chunk_ids": chunks.gather(1, keep),
            "valid": valid.gather(1, keep),
            "chunks_written": int(state["chunks_written"]) + 1,
        }
        self.last_diagnostics = {
            "selected_indices": selected.detach(),
            "selected_token_ids": selected_ids.detach(),
            "selected_scores": selected_scores.detach(),
            "valid_count": new_state["valid"].sum(-1).detach(),
        }
        return new_state

    def _intervene(self, state, intervention):
        values, valid = state["values"], state["valid"]
        if intervention in {"zero", "reset"}:
            return torch.zeros_like(values), torch.zeros_like(valid)
        if intervention == "swap" and values.size(0) > 1:
            return torch.roll(values, 1, 0), torch.roll(valid, 1, 0)
        if intervention == "shuffle":
            return torch.roll(values, 1, 1), torch.roll(valid, 1, 1)
        return values, valid

    def read(self, query_hidden, state, intervention="normal"):
        if state is None:
            context = torch.zeros_like(query_hidden)
            return context, None
        memory, valid = self._intervene(state, intervention)
        q = self.query(query_hidden)
        k = self.key(memory)
        v = self.value(memory)
        scores = torch.einsum("bth,bsh->bts", q.float(), k.float()) / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(~valid[:, None], -torch.inf)
        top_k = min(self.config.reads_per_query, memory.size(1))
        top_scores, top_indices = scores.topk(top_k, dim=-1)
        top_valid = valid[:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices)
        safe_scores = torch.where(top_valid, top_scores, torch.full_like(top_scores, -1e4))
        weights = safe_scores.softmax(-1) * top_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        expanded = v[:, None].expand(-1, query_hidden.size(1), -1, -1)
        selected_values = expanded.gather(
            2, top_indices[..., None].expand(-1, -1, -1, self.hidden_size)
        )
        context = (weights[..., None].to(selected_values.dtype) * selected_values).sum(2)
        context = self.output(context)
        provenance = {
            "slot_indices": top_indices.detach(),
            "weights": weights.detach(),
            "token_ids": state["token_ids"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
            "positions": state["positions"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
            "chunk_ids": state["chunk_ids"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
        }
        return context, provenance

