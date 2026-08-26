"""CPU smoke test for IST v0.3 direct source-token Memory."""
import json
import torch

from config import SourceTokenMemoryConfig
from source_token_memory import SourceTokenMemory


def main():
    torch.manual_seed(3)
    config = SourceTokenMemoryConfig(capacity=8, writes_per_chunk=4, reads_per_query=2, heads=4)
    memory = SourceTokenMemory(16, config)
    hidden = torch.randn(2, 12, 16)
    token_ids = torch.arange(24).reshape(2, 12)
    state = memory.write(hidden, token_ids, chunk_id=0)
    query = torch.randn(2, 3, 16)
    normal, provenance = memory.read(query, state, "normal")
    swapped, _ = memory.read(query, state, "swap")
    result = {
        "status": "pass",
        "state_shape": list(state["values"].shape),
        "valid_per_example": state["valid"].sum(-1).tolist(),
        "source_ids_preserved": bool((state["token_ids"][state["valid"]] >= 0).all()),
        "provenance_shape": list(provenance["token_ids"].shape),
        "swap_changes_context": bool(not torch.allclose(normal, swapped)),
        "trainable_parameters": sum(p.numel() for p in memory.parameters()),
    }
    if not all((result["source_ids_preserved"], result["swap_changes_context"])):
        raise RuntimeError(result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

