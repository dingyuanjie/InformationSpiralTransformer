import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import V05Config
from hybrid_memory import HybridEvidenceCoreMemory
from model import HybridIST
from strict_data import FACT, SCENARIOS, assert_no_leakage, make_batch, split_audit


def tiny_config():
    return V05Config(vocab_size=128, hidden_size=16, heads=4, layers=1, chunk_size=8,
                     evidence_capacity=4, evidence_span=4, writes_per_chunk=2,
                     reads_per_query=2, core_slots=2)


def test_strict_split_shares_vocabulary_but_not_bindings():
    assert_no_leakage(); report = split_audit()
    assert report["binding_overlap"] == 0
    assert report["shared_entity_vocabulary"] and report["shared_value_vocabulary"]


def test_writer_preserves_span_and_provenance():
    config = tiny_config(); memory = HybridEvidenceCoreMemory(config)
    tokens = torch.tensor([[FACT, 16, 3, 48, 90, 91, 92, 93]])
    hidden = torch.randn(1, 8, 16)
    mask = tokens.unfold(1, 4, 1)[:, :, 0].eq(FACT)
    state = memory.write(hidden, tokens, candidate_mask=mask, chunk_id=3, position_offset=24)
    valid = state["evidence"]["valid"][0]
    assert valid.sum() == 1
    assert torch.equal(state["evidence"]["token_ids"][0, valid][0], tokens[0, :4])
    assert state["evidence"]["source_chunks"][0, valid].item() == 3
    assert state["evidence"]["positions"][0, valid][0, 0].item() == 24


def test_read_paths_and_interventions_are_separable():
    config = tiny_config(); model = HybridIST(config)
    batch = make_batch(2, 2, config.chunk_size, 10, "train", facts_per_chunk=2)
    normal, state = model(batch.history, batch.query)
    zero, _ = model(batch.history, batch.query, state, "zero")
    no_evidence, _ = model(batch.history, batch.query, state, "zero_evidence")
    no_core, _ = model(batch.history, batch.query, state, "zero_core")
    assert normal.shape == zero.shape == (2, config.vocab_size)
    assert not torch.allclose(normal, zero)
    assert not torch.allclose(no_evidence, no_core)
    assert model.last_provenance is not None


def test_source_deletion_removes_target_chunk_from_read_candidates():
    config = tiny_config(); model = HybridIST(config)
    batch = make_batch(2, 2, config.chunk_size, 11, "train", facts_per_chunk=2)
    state = model.build_state(batch.history)
    model(batch.history, batch.query, state, "delete_source", source_chunk=0)
    assert not model.last_provenance["source_chunks"].eq(0).any()


def test_query_is_not_required_to_build_history_state():
    config = tiny_config(); model = HybridIST(config)
    batch = make_batch(2, 2, config.chunk_size, 12, "train", facts_per_chunk=2)
    first = model.build_state(batch.history)
    second = model.build_state(batch.history)
    assert torch.allclose(first["evidence"]["values"], second["evidence"]["values"])


def test_all_registered_scenarios_generate_without_binding_leakage():
    for index, scenario in enumerate(SCENARIOS):
        batch = make_batch(2, 4, 16, 100 + index, "strict", scenario=scenario)
        assert batch.history.shape == (2, 4, 16)
        assert batch.query.shape == (2, 3)
