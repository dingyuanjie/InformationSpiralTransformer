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
        # Selection starts from the deterministic novelty signal. A randomly
        # initialized scorer can suppress the answer span before it has learned
        # anything, making provenance depend on initialization noise.
        nn.init.zeros_(self.salience.weight)
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

    def _balanced_keep(self, scores, chunks, valid):
        """Keep a score-ranked quota from every represented source chunk.

        A single global top-k silently turns Memory into a recency/salience race:
        an early fact can disappear merely because later chunks contain many
        moderately salient tokens. The discrete quota is deliberately simple
        and auditable. Unused quota is filled by global score.
        """
        batch, candidates = scores.shape
        capacity = self.config.capacity
        keep = torch.zeros(batch, capacity, dtype=torch.long, device=scores.device)
        for row in range(batch):
            valid_indices = torch.where(valid[row])[0]
            chunk_values = torch.unique(chunks[row, valid_indices], sorted=True)
            if chunk_values.numel() > capacity:
                # More chunks than slots: retain the chunks whose best evidence
                # is strongest. This case is explicit rather than accidental.
                best = torch.stack([
                    scores[row, valid_indices[chunks[row, valid_indices] == chunk]].max()
                    for chunk in chunk_values
                ])
                chunk_values = chunk_values[best.topk(capacity).indices]
            quota = max(1, capacity // max(1, chunk_values.numel()))
            chosen = []
            for chunk in chunk_values:
                group = valid_indices[chunks[row, valid_indices] == chunk]
                count = min(quota, group.numel())
                chosen.extend(group[scores[row, group].topk(count).indices].tolist())
            selected = torch.zeros(candidates, dtype=torch.bool, device=scores.device)
            if chosen:
                selected[torch.tensor(chosen, device=scores.device)] = True
            remaining = valid_indices[~selected[valid_indices]]
            room = capacity - len(chosen)
            if room and remaining.numel():
                chosen.extend(remaining[scores[row, remaining].topk(min(room, remaining.numel())).indices].tolist())
            if len(chosen) < capacity:
                invalid = torch.where(~valid[row])[0].tolist()
                chosen.extend(invalid[:capacity - len(chosen)])
            keep[row] = torch.tensor(chosen[:capacity], device=scores.device)
        return keep

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
        keep = self._balanced_keep(merged_scores, chunks, valid)
        keep_scores = merged_scores.gather(1, keep)
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
        metadata = {key: state[key] for key in ("token_ids", "positions", "chunk_ids")}
        if intervention in {"zero", "reset"}:
            empty = {key: torch.full_like(value, -1) for key, value in metadata.items()}
            return torch.zeros_like(values), torch.zeros_like(valid), empty
        if intervention == "swap" and values.size(0) > 1:
            return (
                torch.roll(values, 1, 0), torch.roll(valid, 1, 0),
                {key: torch.roll(value, 1, 0) for key, value in metadata.items()},
            )
        if intervention == "shuffle":
            return (
                torch.roll(values, 1, 1), torch.roll(valid, 1, 1),
                {key: torch.roll(value, 1, 1) for key, value in metadata.items()},
            )
        return values, valid, metadata

    def read(self, query_hidden, state, intervention="normal"):
        if state is None:
            context = torch.zeros_like(query_hidden)
            return context, None
        memory, valid, metadata = self._intervene(state, intervention)
        q = self.query(query_hidden)
        k = self.key(memory)
        v = self.value(memory)
        scores = torch.einsum("bth,bsh->bts", q.float(), k.float()) / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(~valid[:, None], -torch.inf)
        # Keep a live (non-detached) score tensor for supervised Reader
        # alignment. Public provenance below remains detached diagnostics.
        self.last_read_scores = scores
        self.last_read_positions = metadata["positions"]
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
            "token_ids": metadata["token_ids"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
            "positions": metadata["positions"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
            "chunk_ids": metadata["chunk_ids"][:, None].expand(-1, query_hidden.size(1), -1).gather(2, top_indices).detach(),
        }
        return context, provenance
