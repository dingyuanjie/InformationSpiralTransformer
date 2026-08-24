"""Learned Fast/Slow/Episodic internal Memory with lifecycle diagnostics."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HierarchicalMemoryConfig
from spiral_memory import SpiralMemory


def _cosine_summary(memory: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(memory.float().mean(1), target.float(), dim=-1)


class MemoryRouter(nn.Module):
    def __init__(self, hidden_size: int, config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, 4)
        )

    def forward(self, hidden, fast, slow):
        batch = hidden.size(0)
        if not self.config.enabled or self.config.mode == "disabled":
            routing = hidden.new_zeros(batch, 4); routing[:, 0] = 1.0
            return routing
        features = torch.cat((hidden.mean(1), fast.mean(1), slow.mean(1)), dim=-1)
        probabilities = F.softmax(self.network(features) / self.config.temperature, dim=-1)
        if self.config.mode == "hard_straight_through":
            hard = F.one_hot(probabilities.argmax(-1), 4).to(probabilities.dtype)
            return hard + probabilities - probabilities.detach()
        return probabilities


class HierarchicalMemory(nn.Module):
    """One layer of fixed-capacity hierarchical Memory.

    State is an explicit dictionary so it can be detached, checkpointed, rolled,
    swapped, frozen, and inspected without hidden mutable buffers.
    """

    def __init__(self, hidden_size: int, config: HierarchicalMemoryConfig):
        super().__init__()
        config.validate(hidden_size)
        self.hidden_size = hidden_size
        self.config = config
        self.fast_writer = SpiralMemory(hidden_size, config.fast.slots)
        self.router = MemoryRouter(hidden_size, config.router)
        self.fast_read = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.fast_fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())

        self.slow_queries = nn.Parameter(torch.randn(config.slow.slots, hidden_size) / hidden_size**0.5)
        self.slow_candidate = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.slow_write_gate = nn.Sequential(nn.Linear(hidden_size * 2 + 2, hidden_size), nn.Sigmoid())
        self.slow_retention_gate = nn.Sequential(nn.Linear(hidden_size + 2, hidden_size), nn.Sigmoid())
        nn.init.constant_(self.slow_retention_gate[0].bias, config.slow.retention_bias)
        self.slow_read = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.slow_fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())

        self.consolidation_score = nn.Linear(hidden_size + 3, 1)
        self.episodic_key = nn.Linear(hidden_size, hidden_size)
        self.episodic_value = nn.Linear(hidden_size, hidden_size)
        self.episodic_query = nn.Linear(hidden_size, hidden_size)
        self.episodic_fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())

        self.intervention = "normal"
        self.roll_slots = 1
        self.last_diagnostics = None

    def initialize_state(self, batch: int, device, dtype):
        def zeros(slots): return torch.zeros(batch, slots, self.hidden_size, device=device, dtype=dtype)
        e = self.config.episodic.slots
        return {
            "fast": zeros(self.config.fast.slots), "slow": zeros(self.config.slow.slots),
            "episodic_keys": zeros(e), "episodic_values": zeros(e),
            "episodic_usage": torch.zeros(batch, e, device=device),
            "episodic_age": torch.zeros(batch, e, device=device),
            "episodic_importance": torch.zeros(batch, e, device=device),
            "episodic_occupied": torch.zeros(batch, e, device=device, dtype=torch.bool),
            "fast_usage": torch.zeros(batch, self.config.fast.slots, device=device),
            "fast_age": torch.zeros(batch, self.config.fast.slots, device=device),
            "slow_usage": torch.zeros(batch, self.config.slow.slots, device=device),
            "slow_age": torch.zeros(batch, self.config.slow.slots, device=device),
            "initial_fast": zeros(self.config.fast.slots),
            "initial_slow": zeros(self.config.slow.slots),
            "initial_episodic": zeros(e), "step": 0,
        }

    @staticmethod
    def detach_state(state):
        return {key: value.detach() if torch.is_tensor(value) else value for key, value in state.items()}

    def _route_mode(self, tensor, kind: str):
        intervention = self.intervention
        if intervention in {f"zero_{kind}"} or (intervention.startswith("keep_only_") and
                                                 intervention != f"keep_only_{kind}"):
            return torch.zeros_like(tensor)
        if intervention == f"roll_{kind}":
            return torch.roll(tensor, self.roll_slots, dims=1)
        if intervention == f"swap_{kind}" and tensor.size(0) > 1:
            return torch.roll(tensor, 1, dims=0)
        return tensor

    def _episodic_update(self, hidden_summary, state, route):
        keys, values = state["episodic_keys"], state["episodic_values"]
        usage, age = state["episodic_usage"], state["episodic_age"]
        importance, occupied = state["episodic_importance"], state["episodic_occupied"]
        candidate_key = F.normalize(self.episodic_key(hidden_summary), dim=-1)
        candidate_value = self.episodic_value(hidden_summary)
        similarity = torch.einsum("beh,bh->be", F.normalize(keys.float(), dim=-1),
                                  candidate_key.float()).abs()
        c = self.config.episodic
        eviction = (c.age_weight * age - c.usage_weight * usage -
                    c.importance_weight * importance + c.redundancy_weight * similarity)
        eviction = torch.where(occupied, eviction, torch.full_like(eviction, 1e6))
        selected = eviction.argmax(-1)
        one_hot = F.one_hot(selected, c.slots).to(values.dtype)
        strength = route[:, None] * one_hot
        keys = keys * (1 - strength[..., None]) + candidate_key[:, None] * strength[..., None]
        values = values * (1 - strength[..., None]) + candidate_value[:, None] * strength[..., None]
        age = (age + occupied.float()) * (1 - one_hot.float())
        usage = usage * (1 - one_hot.float())
        importance = importance * (1 - one_hot.float()) + route[:, None] * one_hot.float()
        occupied = occupied | one_hot.bool()
        return keys, values, usage, age, importance, occupied, selected

    def forward(self, hidden, state=None):
        batch, _, _ = hidden.shape
        if state is None:
            state = self.initialize_state(batch, hidden.device, hidden.dtype)
        old_fast, old_slow = state["fast"], state["slow"]
        routing = self.router(hidden, old_fast, old_slow)
        p_fast, p_slow, p_episode, _ = routing.unbind(-1)
        if not self.config.fast.enabled: p_fast = torch.zeros_like(p_fast)
        if not self.config.slow.enabled: p_slow = torch.zeros_like(p_slow)
        if not self.config.episodic.enabled: p_episode = torch.zeros_like(p_episode)

        proposed_fast, base_feature = self.fast_writer(hidden, old_fast)
        new_fast = old_fast + p_fast[:, None, None] * (proposed_fast - old_fast)
        fast_context, fast_weights = self.fast_read(hidden, new_fast, new_fast, need_weights=True)
        fast_gate = self.fast_fusion(torch.cat((hidden, fast_context), dim=-1))
        fast_usage_now = fast_weights.float().mean(1)
        fast_usage = 0.95 * state["fast_usage"] + fast_usage_now
        fast_age = state["fast_age"] + 1

        metrics = torch.stack((state["fast_usage"], state["fast_age"] / 1000,
                               p_fast[:, None].expand_as(state["fast_usage"])), dim=-1)
        consolidation_logits = self.consolidation_score(torch.cat((new_fast.float(), metrics), dim=-1)).squeeze(-1)
        consolidation = torch.sigmoid(consolidation_logits)
        consolidation = consolidation if self.config.consolidation.enabled else torch.zeros_like(consolidation)
        consolidated = (consolidation[..., None] * new_fast.float()).sum(1) / consolidation.sum(1, keepdim=True).clamp_min(1e-6)
        slow_source = torch.cat((hidden, consolidated[:, None].to(hidden.dtype)), dim=1)
        queries = self.slow_queries[None].expand(batch, -1, -1).to(hidden.dtype)
        slow_candidate, _ = self.slow_candidate(queries, slow_source, slow_source, need_weights=False)
        slow_usage_feature = state["slow_usage"][..., None].to(hidden.dtype)
        slow_age_feature = (state["slow_age"] / 1000)[..., None].to(hidden.dtype)
        retention = self.slow_retention_gate(torch.cat((old_slow, slow_usage_feature, slow_age_feature), dim=-1))
        routing_features = torch.stack((p_slow, consolidation.mean(-1)), dim=-1)
        routing_features = routing_features[:, None].expand(-1, old_slow.size(1), -1).to(hidden.dtype)
        write_gate = self.slow_write_gate(torch.cat((old_slow, slow_candidate, routing_features), dim=-1))
        interval_open = ((int(state["step"]) + 1) % max(1, self.config.slow.update_interval) == 0)
        effective = (1 - retention) * write_gate * p_slow[:, None, None] * float(interval_open)
        new_slow = old_slow + effective * (slow_candidate - old_slow)
        slow_context, slow_weights = self.slow_read(hidden, new_slow, new_slow, need_weights=True)
        slow_gate = self.slow_fusion(torch.cat((hidden, slow_context), dim=-1))
        slow_usage_now = slow_weights.float().mean(1)
        slow_usage = 0.98 * state["slow_usage"] + slow_usage_now
        slow_age = state["slow_age"] + (effective.float().mean(-1) < 0.01).float()

        (episode_keys, episode_values, episode_usage, episode_age,
         episode_importance, episode_occupied, replaced) = self._episodic_update(
             hidden.mean(1), state, p_episode if self.config.episodic.enabled else torch.zeros_like(p_episode))
        query = self.episodic_query(hidden)
        scores = torch.einsum("bth,beh->bte", query.float(), episode_keys.float()) / self.hidden_size**0.5
        scores = scores.masked_fill(~episode_occupied[:, None], -1e4)
        top_scores, top_indices = scores.topk(self.config.episodic.top_k, dim=-1)
        top_values = episode_values[:, None].expand(-1, hidden.size(1), -1, -1).gather(
            2, top_indices[..., None].expand(-1, -1, -1, self.hidden_size))
        top_valid = episode_occupied[:, None].expand(-1, hidden.size(1), -1).gather(2, top_indices)
        top_weights = F.softmax(top_scores, dim=-1) * top_valid.float()
        top_weights = top_weights / top_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        episode_context = (top_weights[..., None].to(top_values.dtype) * top_values).sum(2)
        episode_gate = self.episodic_fusion(torch.cat((hidden, episode_context), dim=-1))
        episode_usage = episode_usage.scatter_add(1, top_indices.reshape(batch, -1),
                                                   top_weights.reshape(batch, -1).float())

        if self.intervention == "freeze_fast": new_fast = old_fast
        if self.intervention == "freeze_slow": new_slow = old_slow
        if self.intervention == "freeze_episodic":
            episode_keys, episode_values = state["episodic_keys"], state["episodic_values"]
        read_fast = self._route_mode(new_fast, "fast")
        read_slow = self._route_mode(new_slow, "slow")
        read_episode_keys = self._route_mode(episode_keys, "episodic")
        read_episode = self._route_mode(episode_values, "episodic")
        # Recompute causal reads after every intervention, including freeze.
        fast_context, _ = self.fast_read(hidden, read_fast, read_fast, need_weights=False)
        slow_context, _ = self.slow_read(hidden, read_slow, read_slow, need_weights=False)
        causal_scores = torch.einsum("bth,beh->bte", query.float(), read_episode_keys.float()) / self.hidden_size**0.5
        causal_scores = causal_scores.masked_fill(~episode_occupied[:, None], -1e4)
        causal_top_scores, causal_top_indices = causal_scores.topk(self.config.episodic.top_k, dim=-1)
        selected_values = read_episode[:, None].expand(-1, hidden.size(1), -1, -1).gather(
            2, causal_top_indices[..., None].expand(-1, -1, -1, self.hidden_size))
        causal_valid = episode_occupied[:, None].expand(-1, hidden.size(1), -1).gather(2, causal_top_indices)
        causal_weights = F.softmax(causal_top_scores, dim=-1) * causal_valid.float()
        causal_weights = causal_weights / causal_weights.sum(-1, keepdim=True).clamp_min(1e-6)
        episode_context = (causal_weights[..., None].to(selected_values.dtype) * selected_values).sum(2)

        feature = base_feature if self.config.fast.enabled else hidden
        if self.config.fast.enabled: feature = feature + p_fast[:, None, None] * fast_gate * fast_context
        if self.config.slow.enabled: feature = feature + p_slow[:, None, None] * slow_gate * slow_context
        if self.config.episodic.enabled: feature = feature + p_episode[:, None, None] * episode_gate * episode_context
        new_state = {"fast": new_fast, "slow": new_slow,
                     "episodic_keys": episode_keys, "episodic_values": episode_values,
                     "episodic_usage": episode_usage, "episodic_age": episode_age,
                     "episodic_importance": episode_importance, "episodic_occupied": episode_occupied,
                     "fast_usage": fast_usage, "fast_age": fast_age,
                     "slow_usage": slow_usage, "slow_age": slow_age,
                     "initial_fast": state["initial_fast"] if state["step"] else new_fast.detach(),
                     "initial_slow": state["initial_slow"] if state["step"] else new_slow.detach(),
                     "initial_episodic": state["initial_episodic"] if state["step"] else episode_values.detach(),
                     "step": int(state["step"]) + 1}
        target = hidden.mean(1)
        self.last_diagnostics = {
            "fast_memory_norm": new_fast.float().norm(dim=-1).mean().detach(),
            "slow_memory_norm": new_slow.float().norm(dim=-1).mean().detach(),
            "episodic_memory_norm": episode_values.float().norm(dim=-1).mean().detach(),
            "fast_write_rate": p_fast.mean().detach(),
            "slow_write_rate": effective.float().mean().detach(),
            "episodic_write_rate": p_episode.mean().detach(),
            "retention_gate": retention.float().mean().detach(),
            "importance_score": (p_slow + p_episode).mean().detach(),
            "router_distribution": routing.float().mean(0).detach(),
            "slot_usage": {"fast": fast_usage.detach(), "slow": slow_usage.detach(),
                           "episodic": episode_usage.detach()},
            "slot_age": {"fast": fast_age.detach(), "slow": slow_age.detach(),
                         "episodic": episode_age.detach()},
            "slot_replacement": replaced.detach(), "episodic_topk_usage": top_indices.detach(),
            "memory_similarity_to_initial": torch.stack((
                _cosine_summary(new_fast, new_state["initial_fast"].mean(1)),
                _cosine_summary(new_slow, new_state["initial_slow"].mean(1)),
                _cosine_summary(episode_values, new_state["initial_episodic"].mean(1))), dim=-1).detach(),
            "memory_similarity_to_target_encoding": torch.stack((
                _cosine_summary(new_fast, target), _cosine_summary(new_slow, target),
                _cosine_summary(episode_values, target)), dim=-1).detach(),
            "consolidation_score": consolidation.detach(),
            "fast_read_strength": fast_usage_now.detach(),
            "slow_read_strength": slow_usage_now.detach(),
        }
        return new_state, feature
