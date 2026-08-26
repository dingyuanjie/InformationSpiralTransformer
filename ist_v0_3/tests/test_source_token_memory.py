import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SourceTokenMemoryConfig
from source_token_memory import SourceTokenMemory


def build():
    torch.manual_seed(7)
    return SourceTokenMemory(16, SourceTokenMemoryConfig(
        capacity=6, writes_per_chunk=3, reads_per_query=2, heads=4
    ))


def test_capacity_and_provenance_are_exact():
    module = build()
    hidden = torch.randn(2, 10, 16)
    ids = torch.arange(20).reshape(2, 10)
    state = module.write(hidden, ids, chunk_id=4, position_offset=100)
    assert state["values"].shape == (2, 6, 16)
    valid = state["valid"]
    assert valid.sum(-1).tolist() == [3, 3]
    assert (state["chunk_ids"][valid] == 4).all()
    assert (state["positions"][valid] >= 100).all()
    assert (state["token_ids"][valid] >= 0).all()


def test_read_reports_source_tokens_and_swap_is_causal():
    module = build()
    hidden = torch.randn(2, 10, 16)
    ids = torch.arange(20).reshape(2, 10)
    state = module.write(hidden, ids)
    query = torch.randn(2, 2, 16)
    normal, provenance = module.read(query, state)
    swapped, _ = module.read(query, state, "swap")
    assert provenance["token_ids"].shape == (2, 2, 2)
    assert not torch.allclose(normal, swapped)


def test_zero_intervention_removes_context():
    module = build()
    state = module.write(torch.randn(1, 8, 16), torch.arange(8)[None])
    context, _ = module.read(torch.randn(1, 2, 16), state, "zero")
    assert torch.equal(context, torch.zeros_like(context))

