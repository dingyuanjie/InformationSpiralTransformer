import torch

from ist_v0_2.config import HierarchicalMemoryConfig
from ist_v0_2.model import build_model


def build(config=None):
    return build_model("hierarchical_v0_2", vocab_size=19, hidden_size=64, layers=2,
                       max_sequence_length=32, hierarchical_config=config or HierarchicalMemoryConfig())


def test_shapes_and_backward():
    model = build(); tokens = torch.randint(19, (2, 32))
    logits, state = model(tokens, return_memory=True)
    assert logits.shape == (2, 32, 19)
    assert state[0]["fast"].shape == (2, 32, 64)
    assert state[0]["slow"].shape == (2, 8, 64)
    assert state[0]["episodic_values"].shape == (2, 64, 64)
    (logits.mean() + model.memory_diversity_loss()).backward()


def test_cross_chunk_and_interventions_are_finite():
    model = build().eval(); tokens = torch.randint(19, (2, 32))
    with torch.no_grad(): _, state = model(tokens, return_memory=True)
    for intervention in ("zero_fast", "zero_slow", "zero_episodic", "freeze_fast",
                         "freeze_slow", "freeze_episodic", "keep_only_fast",
                         "keep_only_slow", "keep_only_episodic", "roll_fast",
                         "roll_slow", "roll_episodic", "swap_fast", "swap_slow",
                         "swap_episodic"):
        model.set_memory_intervention(intervention)
        with torch.no_grad(): logits, _ = model(tokens, memory=state, return_memory=True)
        assert torch.isfinite(logits).all()


def test_all_components_can_be_disabled():
    for component in ("fast", "slow", "episodic", "router", "consolidation"):
        config = HierarchicalMemoryConfig().to_dict(); config[component]["enabled"] = False
        model = build(config).eval()
        with torch.no_grad(): logits = model(torch.randint(19, (1, 32)))
        assert torch.isfinite(logits).all()


def test_bfloat16_forward_without_autocast():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return
    memory = build().blocks[0].memory.cuda().to(torch.bfloat16).eval()
    hidden = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        state, feature = memory(hidden)
        _, feature = memory(hidden, state)
    assert feature.dtype == torch.bfloat16
    assert torch.isfinite(feature).all()
