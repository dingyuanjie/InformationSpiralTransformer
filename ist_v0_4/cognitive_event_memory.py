"""Working, episodic and semantic event memory for IST v0.4."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CognitiveMemoryConfig


class CognitiveEventMemory(nn.Module):
    """Store complete source spans and forget by explicit lifecycle utility.

    Working memory is a recency FIFO. Episodic memory retains exact source spans
    selected by surprise and novelty, then evicts weak, old, unused traces.
    Repeatedly retrieved episodes can be consolidated into semantic prototypes.
    """

    def __init__(self, hidden_size: int, config: CognitiveMemoryConfig | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config or CognitiveMemoryConfig()
        self.config.validate(hidden_size)
        self.event_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.semantic_output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_diagnostics = None

    def _store(self, batch, capacity, device, dtype):
        span, hidden = self.config.event_span, self.hidden_size
        return {
            "values": torch.zeros(batch, capacity, span, hidden, device=device, dtype=dtype),
            "token_ids": torch.full((batch, capacity, span), -1, device=device, dtype=torch.long),
            "positions": torch.full((batch, capacity, span), -1, device=device, dtype=torch.long),
            "token_valid": torch.zeros(batch, capacity, span, device=device, dtype=torch.bool),
            "keys": torch.zeros(batch, capacity, hidden, device=device, dtype=dtype),
            "strength": torch.zeros(batch, capacity, device=device),
            "born": torch.full((batch, capacity), -1, device=device, dtype=torch.long),
            "last_access": torch.full((batch, capacity), -1, device=device, dtype=torch.long),
            "accesses": torch.zeros(batch, capacity, device=device, dtype=torch.long),
            "valid": torch.zeros(batch, capacity, device=device, dtype=torch.bool),
        }

    def empty_state(self, batch, device, dtype):
        return {
            "working": self._store(batch, self.config.working_events, device, dtype),
            "episodic": self._store(batch, self.config.episodic_events, device, dtype),
            "semantic": {
                "keys": torch.zeros(batch, self.config.semantic_slots, self.hidden_size, device=device, dtype=dtype),
                "counts": torch.zeros(batch, self.config.semantic_slots, device=device),
                "valid": torch.zeros(batch, self.config.semantic_slots, device=device, dtype=torch.bool),
                "source_token_ids": torch.full(
                    (batch, self.config.semantic_slots, self.config.event_span), -1,
                    device=device, dtype=torch.long),
                "source_positions": torch.full(
                    (batch, self.config.semantic_slots, self.config.event_span), -1,
                    device=device, dtype=torch.long),
                "source_valid": torch.zeros(
                    batch, self.config.semantic_slots, self.config.event_span,
                    device=device, dtype=torch.bool),
            },
            "clock": 0,
        }

    @staticmethod
    def detach_state(state):
        result = {}
        for key, value in state.items():
            if isinstance(value, dict):
                result[key] = CognitiveEventMemory.detach_state(value)
            else:
                result[key] = value.detach() if torch.is_tensor(value) else value
        return result

    def _events(self, hidden, token_ids, position_offset):
        batch, tokens, hidden_size = hidden.shape
        span, stride = self.config.event_span, self.config.event_stride
        starts = torch.arange(0, tokens, stride, device=hidden.device)
        indices = starts[:, None] + torch.arange(span, device=hidden.device)[None]
        valid_indices = indices < tokens
        safe = indices.clamp_max(tokens - 1)
        count = starts.numel()
        values = hidden[:, safe].masked_fill(~valid_indices[None, :, :, None], 0)
        ids = token_ids[:, safe].masked_fill(~valid_indices[None], -1)
        valid = ids >= 0
        positions = indices.add(position_offset).reshape(1, count, span).expand(batch, -1, -1)
        positions = positions.masked_fill(~valid, -1)
        denominator = valid.sum(-1, keepdim=True).clamp_min(1)
        pooled = (values.float() * valid[..., None]).sum(2) / denominator
        keys = self.event_key(pooled.to(self.event_key.weight.dtype)).to(hidden.dtype)
        residual = values.float() - pooled[:, :, None]
        surprise = (residual.norm(dim=-1) * valid).sum(-1) / denominator.squeeze(-1)
        return values, ids, positions, valid, keys, surprise

    def _novelty(self, keys, episodic):
        if not episodic["valid"].any():
            return torch.ones(keys.shape[:2], device=keys.device)
        similarity = torch.einsum(
            "beh,bsh->bes", F.normalize(keys.float(), dim=-1),
            F.normalize(episodic["keys"].float(), dim=-1),
        ).masked_fill(~episodic["valid"][:, None], -1)
        return 1 - similarity.max(-1).values.clamp(-1, 1)

    @staticmethod
    def _gather_store(store, keep):
        result = {}
        for key, value in store.items():
            if value.ndim == 4:
                index = keep[..., None, None].expand(-1, -1, value.size(2), value.size(3))
            elif value.ndim == 3:
                index = keep[..., None].expand(-1, -1, value.size(2))
            else:
                index = keep
            result[key] = value.gather(1, index)
        return result

    def _append_and_keep(self, store, event, capacity, utility):
        merged = {key: torch.cat((store[key], event[key]), dim=1) for key in store}
        merged_utility = torch.cat((utility[0], utility[1]), dim=1)
        merged_utility = merged_utility.masked_fill(~merged["valid"], -torch.inf)
        keep = merged_utility.topk(capacity, dim=1).indices
        return self._gather_store(merged, keep)

    def write(self, hidden, token_ids, state=None, chunk_id=0, position_offset=0):
        if state is None:
            state = self.empty_state(hidden.size(0), hidden.device, hidden.dtype)
        clock = int(state["clock"]) + 1
        values, ids, positions, token_valid, keys, surprise = self._events(hidden, token_ids, position_offset)
        novelty = self._novelty(keys, state["episodic"])
        score = self.config.surprise_weight * surprise + self.config.novelty_weight * novelty
        batch, events = score.shape
        common = {
            "values": values, "token_ids": ids, "positions": positions,
            "token_valid": token_valid, "keys": keys,
            "strength": score.float(),
            "born": torch.full((batch, events), clock, device=hidden.device, dtype=torch.long),
            "last_access": torch.full((batch, events), clock, device=hidden.device, dtype=torch.long),
            "accesses": torch.zeros(batch, events, device=hidden.device, dtype=torch.long),
            "valid": token_valid.any(-1),
        }
        # Working memory keeps the newest complete events, independent of score.
        old_working_utility = state["working"]["born"].float()
        new_working_utility = common["born"].float()
        working = self._append_and_keep(state["working"], common, self.config.working_events,
                                        (old_working_utility, new_working_utility))
        admitted = min(self.config.admissions_per_chunk, events)
        selected = score.topk(admitted, dim=1).indices
        selected_event = self._gather_store(common, selected)
        episodic = state["episodic"]
        age = (clock - episodic["born"]).clamp_min(0).float()
        idle = (clock - episodic["last_access"]).clamp_min(0).float()
        old_utility = (episodic["strength"] + self.config.access_bonus * torch.log1p(episodic["accesses"].float())
                       - self.config.age_decay * (age + 0.5 * idle))
        episodic_new = self._append_and_keep(episodic, selected_event, self.config.episodic_events,
                                             (old_utility, selected_event["strength"]))
        new_state = {"working": working, "episodic": episodic_new,
                     "semantic": state["semantic"], "clock": clock}
        self.last_diagnostics = {"event_scores": score.detach(), "novelty": novelty.detach(),
                                 "surprise": surprise.detach(), "admitted": selected.detach()}
        return new_state

    def read(self, query_hidden, state, intervention="normal"):
        if state is None or intervention in {"zero", "reset"}:
            return torch.zeros_like(query_hidden), None
        stores = []
        if intervention != "zero_working": stores.append((0, state["working"]))
        if intervention != "zero_episodic": stores.append((1, state["episodic"]))
        valid = torch.cat([store["valid"] for _, store in stores], dim=1)
        values = torch.cat([store["values"] for _, store in stores], dim=1)
        token_valid = torch.cat([store["token_valid"] for _, store in stores], dim=1)
        token_ids = torch.cat([store["token_ids"] for _, store in stores], dim=1)
        positions = torch.cat([store["positions"] for _, store in stores], dim=1)
        store_ids = torch.cat([torch.full_like(store["valid"], kind) for kind, store in stores], dim=1)
        if intervention == "swap" and values.size(0) > 1:
            values, token_valid, token_ids, positions, valid, store_ids = [
                torch.roll(item, 1, 0)
                for item in (values, token_valid, token_ids, positions, valid, store_ids)
            ]
        denominator = token_valid.sum(-1, keepdim=True).clamp_min(1)
        pooled = (values.float() * token_valid[..., None]).sum(2) / denominator
        # Recompute retrieval keys from exact stored events. This lets the Key
        # projection learn without keeping a graph through every source chunk.
        keys = self.event_key(pooled.to(self.event_key.weight.dtype)).to(values.dtype)
        q = self.query(query_hidden)
        event_scores = torch.einsum("bth,bsh->bts", q.float(), keys.float()) / math.sqrt(self.hidden_size)
        event_scores = event_scores.masked_fill(~valid[:, None], -torch.inf)
        self.last_event_scores = event_scores
        self.last_event_positions = positions
        self.last_event_token_ids = token_ids
        self.last_event_store_ids = store_ids
        count = min(self.config.retrieved_events, keys.size(1))
        top_scores, top = event_scores.topk(count, dim=-1)
        top_valid = valid[:, None].expand(-1, query_hidden.size(1), -1).gather(2, top)
        safe = top_scores.masked_fill(~top_valid, -1e4)
        event_weights = safe.softmax(-1) * top_valid.float()
        event_weights /= event_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        gather_values = values[:, None].expand(-1, query_hidden.size(1), -1, -1, -1).gather(
            2, top[..., None, None].expand(-1, -1, -1, self.config.event_span, self.hidden_size))
        gather_valid = token_valid[:, None].expand(-1, query_hidden.size(1), -1, -1).gather(
            2, top[..., None].expand(-1, -1, -1, self.config.event_span))
        token_scores = torch.einsum("bth,btesh->btes", q.float(), gather_values.float())
        token_scores = token_scores.masked_fill(~gather_valid, -1e4)
        token_weights = token_scores.softmax(-1) * gather_valid.float()
        event_context = (token_weights[..., None].to(gather_values.dtype) * self.value(gather_values)).sum(3)
        context = (event_weights[..., None].to(event_context.dtype) * event_context).sum(2)
        context = self.output(context)
        semantic_provenance = None
        if intervention != "zero_semantic" and state["semantic"]["valid"].any():
            semantic_scores = torch.einsum("bth,bsh->bts", q.float(), state["semantic"]["keys"].float())
            semantic_scores = semantic_scores.masked_fill(~state["semantic"]["valid"][:, None], -1e4)
            semantic_weights = semantic_scores.softmax(-1)
            semantic = torch.einsum("bts,bsh->bth", semantic_weights.to(keys.dtype), state["semantic"]["keys"])
            context = context + self.config.semantic_mix * self.semantic_output(semantic)
            semantic_provenance = {
                "weights": semantic_weights.detach(),
                "source_token_ids": state["semantic"]["source_token_ids"].detach(),
                "source_positions": state["semantic"]["source_positions"].detach(),
            }
        expanded_ids = token_ids[:, None].expand(-1, query_hidden.size(1), -1, -1)
        expanded_positions = positions[:, None].expand_as(expanded_ids)
        provenance = {
            "event_indices": top.detach(), "event_weights": event_weights.detach(),
            "store_ids": store_ids[:, None].expand(-1, query_hidden.size(1), -1).gather(2, top).detach(),
            "token_ids": expanded_ids.gather(2, top[..., None].expand(-1, -1, -1, self.config.event_span)).detach(),
            "positions": expanded_positions.gather(2, top[..., None].expand(-1, -1, -1, self.config.event_span)).detach(),
            "token_weights": token_weights.detach(),
            "semantic": semantic_provenance,
        }
        return context, provenance

    def reinforce_from_provenance(self, state, provenance, min_weight=0.6, mode="fixed"):
        """Rehearse confident episodic top-1 events without answer supervision."""
        weights = provenance["event_weights"][:, -1]
        stores = provenance["store_ids"][:, -1]
        indices = provenance["event_indices"][:, -1]
        slots = torch.full((weights.size(0), 1), -1, device=weights.device, dtype=torch.long)
        working_capacity = self.config.working_events
        confident = weights[:, 0] >= min_weight
        episodic = stores[:, 0] == 1
        selected = confident & episodic
        slots[selected, 0] = indices[selected, 0] - working_capacity
        return self.reinforce(state, slots, mode=mode), slots

    def reinforce(self, state, episodic_slots, mode="fixed"):
        """Rehearse retrieved episodic slots and consolidate repeated traces."""
        if mode not in {"fixed", "relative"}:
            raise ValueError("reinforcement mode must be fixed or relative")
        result = self.detach_state(state)
        episodic, semantic = result["episodic"], result["semantic"]
        for batch in range(episodic_slots.size(0)):
            for slot in torch.unique(episodic_slots[batch]).tolist():
                if slot < 0 or slot >= self.config.episodic_events or not episodic["valid"][batch, slot]:
                    continue
                episodic["accesses"][batch, slot] += 1
                episodic["last_access"][batch, slot] = int(result["clock"])
                boost = self.config.access_bonus
                if mode == "relative":
                    boost *= max(1.0, float(episodic["strength"][batch, slot]))
                episodic["strength"][batch, slot] += boost
                if episodic["accesses"][batch, slot] >= self.config.consolidation_accesses:
                    empty = torch.where(~semantic["valid"][batch])[0]
                    target = int(empty[0]) if empty.numel() else int(torch.argmax(
                        F.cosine_similarity(semantic["keys"][batch], episodic["keys"][batch, slot][None], dim=-1)))
                    count = semantic["counts"][batch, target]
                    semantic["keys"][batch, target] = (
                        semantic["keys"][batch, target] * count + episodic["keys"][batch, slot]
                    ) / (count + 1)
                    semantic["counts"][batch, target] = count + 1
                    semantic["valid"][batch, target] = True
                    semantic["source_token_ids"][batch, target] = episodic["token_ids"][batch, slot]
                    semantic["source_positions"][batch, target] = episodic["positions"][batch, slot]
                    semantic["source_valid"][batch, target] = episodic["token_valid"][batch, slot]
        return result
