"""Fixed-capacity, provenance-preserving Evidence + recursive Core memory."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import V05Config


def detach_state(state: dict[str, Any] | None):
    if state is None:
        return None
    return {key: detach_state(value) if isinstance(value, dict)
            else value.detach() if torch.is_tensor(value) else value
            for key, value in state.items()}


class HybridEvidenceCoreMemory(nn.Module):
    """A minimal dual-path memory with exact span provenance."""

    def __init__(self, config: V05Config):
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_size
        self.evidence_query = nn.Linear(hidden, hidden, bias=False)
        self.evidence_key = nn.Linear(hidden, hidden, bias=False)
        self.evidence_value = nn.Linear(hidden, hidden, bias=False)
        self.core_query = nn.Linear(hidden, hidden, bias=False)
        self.core_key = nn.Linear(hidden, hidden, bias=False)
        self.core_value = nn.Linear(hidden, hidden, bias=False)
        self.importance = nn.Linear(hidden, 1)
        self.core_seed = nn.Parameter(torch.randn(config.core_slots, hidden) / math.sqrt(hidden))
        self.core_update_query = nn.Linear(hidden, hidden, bias=False)
        self.core_update_key = nn.Linear(hidden, hidden, bias=False)
        self.core_update_value = nn.Linear(hidden, hidden, bias=False)
        self.core_gate = nn.Linear(hidden * 2, hidden)
        self.core_norm = nn.LayerNorm(hidden)
        self.role_embedding = nn.Parameter(torch.randn(config.evidence_span, hidden) / math.sqrt(hidden))
        self.reranker = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.evidence_gate = nn.Parameter(torch.tensor(config.evidence_gate_init))
        self.core_read_gate = nn.Parameter(torch.tensor(config.core_gate_init))
        self.last_diagnostics: dict[str, Any] = {}

    def empty_state(self, batch: int, device, dtype):
        c = self.config
        evidence = {
            "values": torch.zeros(batch, c.evidence_capacity, c.evidence_span, c.hidden_size,
                                  device=device, dtype=dtype),
            "token_ids": torch.full((batch, c.evidence_capacity, c.evidence_span), -1,
                                    device=device, dtype=torch.long),
            "positions": torch.full((batch, c.evidence_capacity, c.evidence_span), -1,
                                    device=device, dtype=torch.long),
            "source_chunks": torch.full((batch, c.evidence_capacity), -1,
                                        device=device, dtype=torch.long),
            "born": torch.full((batch, c.evidence_capacity), -1, device=device, dtype=torch.long),
            "last_read": torch.full((batch, c.evidence_capacity), -1, device=device, dtype=torch.long),
            "usage": torch.zeros(batch, c.evidence_capacity, device=device),
            "importance": torch.zeros(batch, c.evidence_capacity, device=device),
            "valid": torch.zeros(batch, c.evidence_capacity, device=device, dtype=torch.bool),
        }
        core = self.core_seed.to(device=device, dtype=dtype)[None].expand(batch, -1, -1).clone()
        return {"evidence": evidence, "core": core, "clock": 0}

    @staticmethod
    def _gather(store, indices):
        gathered = {}
        for key, value in store.items():
            suffix = value.shape[2:]
            index = indices.reshape(*indices.shape, *([1] * len(suffix))).expand(*indices.shape, *suffix)
            gathered[key] = value.gather(1, index)
        return gathered

    def _window_candidates(self, hidden, token_ids, chunk_id, position_offset, clock):
        c = self.config
        if hidden.size(1) < c.evidence_span:
            raise ValueError("chunk shorter than evidence_span")
        values = hidden.unfold(1, c.evidence_span, 1).permute(0, 1, 3, 2).contiguous()
        ids = token_ids.unfold(1, c.evidence_span, 1)
        count = values.size(1)
        positions = (torch.arange(count, device=hidden.device)[:, None]
                     + torch.arange(c.evidence_span, device=hidden.device)[None] + position_offset)
        positions = positions[None].expand(hidden.size(0), -1, -1)
        pooled = values.mean(2)
        return {
            "values": values,
            "token_ids": ids,
            "positions": positions,
            "source_chunks": torch.full((hidden.size(0), count), chunk_id,
                                        device=hidden.device, dtype=torch.long),
            "born": torch.full((hidden.size(0), count), clock,
                               device=hidden.device, dtype=torch.long),
            "last_read": torch.full((hidden.size(0), count), clock,
                                    device=hidden.device, dtype=torch.long),
            "usage": torch.zeros(hidden.size(0), count, device=hidden.device),
            "importance": self.importance(pooled).squeeze(-1).float(),
            "valid": torch.ones(hidden.size(0), count, device=hidden.device, dtype=torch.bool),
        }

    def write(self, hidden, token_ids, state=None, chunk_id=0, position_offset=0,
              block_writer=False, candidate_mask=None):
        if state is None:
            state = self.empty_state(hidden.size(0), hidden.device, hidden.dtype)
        if block_writer:
            return state
        clock = int(state["clock"]) + 1
        candidates = self._window_candidates(hidden, token_ids, chunk_id, position_offset, clock)
        if candidate_mask is not None:
            if candidate_mask.shape != candidates["valid"].shape:
                raise ValueError("candidate_mask must cover every sliding window")
            candidates["valid"] = candidate_mask.bool()
        pooled = candidates["values"].mean(2)
        old = state["evidence"]
        if old["valid"].any():
            old_pooled = old["values"].mean(2)
            similarity = torch.einsum("bnh,bkh->bnk", F.normalize(pooled.float(), dim=-1),
                                      F.normalize(old_pooled.float(), dim=-1))
            similarity = similarity.masked_fill(~old["valid"][:, None], -1)
            novelty = 1 - similarity.max(-1).values.clamp(-1, 1)
        else:
            novelty = torch.ones(pooled.shape[:2], device=hidden.device)
        within = torch.einsum("bnh,bmh->bnm", F.normalize(pooled.float(), dim=-1),
                              F.normalize(pooled.float(), dim=-1))
        diagonal = torch.eye(within.size(-1), device=hidden.device, dtype=torch.bool)[None]
        redundancy = within.masked_fill(diagonal, -1).max(-1).values.clamp_min(0)
        candidate_score = (candidates["importance"] + self.config.novelty_weight * novelty
                           - self.config.redundancy_weight * redundancy)
        candidate_score = candidate_score.masked_fill(~candidates["valid"], -torch.inf)
        choose = candidate_score.topk(self.config.writes_per_chunk, dim=1).indices
        selected = self._gather(candidates, choose)
        selected["importance"] = candidate_score.gather(1, choose)
        age = (clock - old["born"]).clamp_min(0).float()
        old_score = old["importance"] + self.config.usage_bonus * torch.log1p(old["usage"])
        old_score = old_score - self.config.age_decay * age
        merged = {key: torch.cat((old[key], selected[key]), dim=1) for key in old}
        scores = torch.cat((old_score, selected["importance"]), dim=1)
        scores = scores.masked_fill(~merged["valid"], -torch.inf)
        keep = scores.topk(self.config.evidence_capacity, dim=1).indices
        evidence = self._gather(merged, keep)
        core = self._update_core(state["core"], evidence)
        self.last_diagnostics = {
            "candidate_scores": candidate_score.detach(), "candidate_indices": choose.detach(),
            "retained_sources": evidence["source_chunks"].detach(),
            "slot_usage": evidence["valid"].float().mean().detach(),
        }
        return {"evidence": evidence, "core": core, "clock": clock}

    def _update_core(self, core, evidence):
        valid = evidence["valid"]
        summary = evidence["values"].mean(2)
        q, k, v = self.core_update_query(core), self.core_update_key(summary), self.core_update_value(summary)
        scores = torch.einsum("bch,bkh->bck", q.float(), k.float()) / math.sqrt(self.config.hidden_size)
        scores = scores.masked_fill(~valid[:, None], -1e4)
        weights = scores.softmax(-1) * valid[:, None].float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        delta = torch.einsum("bck,bkh->bch", weights.to(v.dtype), v)
        gate = torch.sigmoid(self.core_gate(torch.cat((core, delta), dim=-1)))
        return self.core_norm(core + gate * delta)

    def _intervene(self, state, intervention, source_chunk):
        evidence, core = state["evidence"], state["core"]
        if intervention in {"zero", "reset"}:
            return None, None
        evidence = {key: value for key, value in evidence.items()}
        if intervention == "swap" and core.size(0) > 1:
            evidence = {key: torch.roll(value, 1, 0) for key, value in evidence.items()}
            core = torch.roll(core, 1, 0)
        elif intervention == "shuffle":
            order = torch.arange(self.config.evidence_capacity - 1, -1, -1, device=core.device)[None]
            order = order.expand(core.size(0), -1)
            evidence = self._gather(evidence, order)
        elif intervention == "delete_source":
            if source_chunk is None:
                raise ValueError("delete_source requires source_chunk")
            if torch.is_tensor(source_chunk) and source_chunk.ndim == 1:
                source_chunk = source_chunk[:, None]
            evidence["valid"] = evidence["valid"] & (evidence["source_chunks"] != source_chunk)
        elif intervention in {"swap_entities", "swap_answers", "rebind", "corrupt_identity", "corrupt_roles"}:
            evidence["values"] = evidence["values"].clone()
            evidence["token_ids"] = evidence["token_ids"].clone()
            for batch in range(core.size(0)):
                slots = torch.where(evidence["valid"][batch])[0]
                if slots.numel() < 2:
                    continue
                if intervention in {"swap_entities", "rebind", "corrupt_identity"}:
                    evidence["values"][batch, slots, 1] = torch.roll(
                        evidence["values"][batch, slots, 1].clone(), 1, 0)
                    evidence["token_ids"][batch, slots, 1] = torch.roll(
                        evidence["token_ids"][batch, slots, 1].clone(), 1, 0)
                if intervention in {"swap_answers", "rebind", "corrupt_identity"}:
                    evidence["values"][batch, slots, 3] = torch.roll(
                        evidence["values"][batch, slots, 3].clone(), -1, 0)
                    evidence["token_ids"][batch, slots, 3] = torch.roll(
                        evidence["token_ids"][batch, slots, 3].clone(), -1, 0)
                if intervention == "corrupt_roles":
                    old = evidence["values"][batch, slots].clone()
                    old_ids = evidence["token_ids"][batch, slots].clone()
                    evidence["values"][batch, slots, 1] = old[:, 3]
                    evidence["values"][batch, slots, 3] = old[:, 1]
                    evidence["token_ids"][batch, slots, 1] = old_ids[:, 3]
                    evidence["token_ids"][batch, slots, 3] = old_ids[:, 1]
        return evidence, core

    def read(self, query_hidden, state, intervention="normal", source_chunk=None):
        batch, _, hidden = query_hidden.shape
        if state is None or intervention == "block_reader":
            return torch.zeros(batch, hidden, device=query_hidden.device, dtype=query_hidden.dtype), None
        evidence, core = self._intervene(state, intervention, source_chunk)
        if evidence is None:
            return torch.zeros(batch, hidden, device=query_hidden.device, dtype=query_hidden.dtype), None
        use_evidence = intervention != "zero_evidence"
        use_core = intervention != "zero_core"
        query = self.evidence_query(query_hidden)
        keys = self.evidence_key(evidence["values"])
        token_scores = torch.einsum("bqh,bksh->bqks", query.float(), keys.float()) / math.sqrt(hidden)
        maxsim = token_scores.max(-1).values.sum(1)
        age = (int(state["clock"]) - evidence["born"]).clamp_min(0).float()
        ordered = (evidence["values"] * self.role_embedding[None, None].to(evidence["values"].dtype)).sum(2)
        query_summary = query_hidden.mean(1)[:, None].expand(-1, ordered.size(1), -1)
        rerank = self.reranker(torch.cat((query_summary, ordered), dim=-1)).squeeze(-1).float()
        span_scores = (maxsim + self.config.reranker_weight * rerank) / self.config.reader_temperature
        span_scores = span_scores - self.config.age_decay * age
        span_scores = span_scores.masked_fill(~evidence["valid"], -torch.inf)
        self.last_span_scores = span_scores
        count = min(self.config.reads_per_query, self.config.evidence_capacity)
        top_scores, top = span_scores.topk(count, dim=-1)
        top_valid = evidence["valid"].gather(1, top)
        weights = top_scores.masked_fill(~top_valid, -1e4).softmax(-1) * top_valid.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        chosen_values = evidence["values"].gather(
            1, top[..., None, None].expand(-1, -1, self.config.evidence_span, hidden))
        values = self.evidence_value(chosen_values)
        chosen_token_scores = token_scores.gather(
            2, top[:, None, :, None].expand(-1, query_hidden.size(1), -1, self.config.evidence_span))
        token_weights = chosen_token_scores.max(1).values.softmax(-1)
        span_context = (token_weights[..., None].to(values.dtype) * values).sum(2)
        evidence_context = (weights[..., None].to(values.dtype) * span_context).sum(1)
        qcore = self.core_query(query_hidden.mean(1))
        core_scores = torch.einsum("bh,bch->bc", qcore.float(), self.core_key(core).float()) / math.sqrt(hidden)
        core_weights = core_scores.softmax(-1)
        core_context = torch.einsum("bc,bch->bh", core_weights.to(core.dtype), self.core_value(core))
        context = query_hidden.new_zeros(batch, hidden)
        if use_evidence:
            context = context + torch.sigmoid(self.evidence_gate).to(context.dtype) * evidence_context
        if use_core:
            context = context + torch.sigmoid(self.core_read_gate).to(context.dtype) * core_context
        provenance = {
            "slots": top.detach(), "weights": weights.detach(),
            "source_chunks": evidence["source_chunks"].gather(1, top).detach(),
            "token_ids": evidence["token_ids"].gather(
                1, top[..., None].expand(-1, -1, self.config.evidence_span)).detach(),
            "positions": evidence["positions"].gather(
                1, top[..., None].expand(-1, -1, self.config.evidence_span)).detach(),
            "core_weights": core_weights.detach(),
        }
        if intervention == "normal":
            with torch.no_grad():
                evidence["usage"].scatter_add_(1, top, weights)
                evidence["last_read"].scatter_(1, top, int(state["clock"]))
        self.last_diagnostics.update({"read": provenance})
        return context, provenance
