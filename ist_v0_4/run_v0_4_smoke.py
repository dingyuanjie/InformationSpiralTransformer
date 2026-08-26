"""Smoke test for IST v0.4 cognitive event memory."""
import json
import torch

from cognitive_event_memory import CognitiveEventMemory
from config import CognitiveMemoryConfig


def main():
    torch.manual_seed(404)
    config = CognitiveMemoryConfig(event_span=4, event_stride=4, working_events=2, episodic_events=4,
                                   semantic_slots=2, admissions_per_chunk=1,
                                   retrieved_events=2, consolidation_accesses=2)
    memory = CognitiveEventMemory(16, config)
    state = None
    for chunk in range(3):
        hidden = torch.randn(2, 8, 16)
        ids = torch.arange(chunk * 8, (chunk + 1) * 8).repeat(2, 1)
        state = memory.write(hidden, ids, state, chunk, chunk * 8)
    context, provenance = memory.read(torch.randn(2, 3, 16), state)
    zero, _ = memory.read(torch.randn(2, 3, 16), state, "zero")
    payload = {
        "status": "pass",
        "working_valid": state["working"]["valid"].sum(-1).tolist(),
        "episodic_valid": state["episodic"]["valid"].sum(-1).tolist(),
        "context_shape": list(context.shape),
        "event_provenance_shape": list(provenance["event_indices"].shape),
        "source_span_shape": list(provenance["token_ids"].shape),
        "zero_is_zero": bool(torch.equal(zero, torch.zeros_like(zero))),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
