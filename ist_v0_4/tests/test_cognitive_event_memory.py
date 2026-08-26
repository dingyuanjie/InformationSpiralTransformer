import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cognitive_event_memory import CognitiveEventMemory
from config import CognitiveMemoryConfig


def build():
    torch.manual_seed(4)
    return CognitiveEventMemory(12, CognitiveMemoryConfig(
        event_span=4, working_events=2, episodic_events=4, semantic_slots=2,
        admissions_per_chunk=1, retrieved_events=2, consolidation_accesses=2,
    ))


def test_complete_source_spans_and_provenance_survive():
    memory = build()
    ids = torch.arange(8)[None]
    state = memory.write(torch.randn(1, 8, 12), ids, position_offset=20)
    valid = state["working"]["valid"][0]
    stored = state["working"]["token_ids"][0, valid]
    assert sorted(stored.flatten().tolist()) == list(range(8))
    assert (state["working"]["positions"][0, valid] >= 20).all()


def test_working_memory_forgets_old_events_by_recency():
    memory = build(); state = None
    for chunk in range(3):
        ids = torch.arange(chunk * 8, (chunk + 1) * 8)[None]
        state = memory.write(torch.randn(1, 8, 12), ids, state, chunk, chunk * 8)
    retained = set(state["working"]["token_ids"][0][state["working"]["token_valid"][0]].tolist())
    assert retained == set(range(16, 24))


def test_read_is_causal_and_reports_event_spans():
    memory = build()
    state = memory.write(torch.randn(2, 8, 12), torch.arange(16).reshape(2, 8))
    query = torch.randn(2, 2, 12)
    context, provenance = memory.read(query, state)
    zero, _ = memory.read(query, state, "zero")
    assert provenance["token_ids"].shape == (2, 2, 2, 4)
    assert not torch.equal(context, zero)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_rehearsal_consolidates_repeated_episode():
    memory = build()
    state = memory.write(torch.randn(1, 8, 12), torch.arange(8)[None])
    slot = torch.where(state["episodic"]["valid"][0])[0][:1][None]
    state = memory.reinforce(state, slot)
    state = memory.reinforce(state, slot)
    assert state["semantic"]["valid"].any()
    assert state["episodic"]["accesses"].max() >= 2

