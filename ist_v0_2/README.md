# Information Spiral Transformer v0.2

IST v0.2 is an isolated experimental branch. The sibling `ist_v0_1` directory and all old experiment artifacts remain unchanged. Use `memory_arch="v0_1"` for the exact legacy module/state-dict layout or `memory_arch="hierarchical_v0_2"` for the lifecycle prototype.

## Data flow

```text
Current Chunk
  -> soft / straight-through Memory Router
     -> Fast Memory [B, 32, H]       frequent, short-lived state
     -> Slow Memory [B, 8, H]        protected learned retention/write gates
     -> Episodic keys/values [B,64,H] finite detail store, top-k=4 retrieval
     -> Forget route                 suppresses writes

Fast slots + usage/write/read diagnostics
  -> learned consolidation score
  -> Slow candidate

Fast read + Slow read + Episodic top-k read
  -> independent fusion gates
  -> block FFN / output
```

All three memories are explicit per-layer state dictionaries and can cross Chunk boundaries, detach, checkpoint, roll, swap, zero, freeze, or be isolated with `keep_only_*` interventions.

## Commands

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_level8_0_local.py --dry-run
python run_level8_0_local.py --smoke-test
python run_level8_1_local.py --dry-run
python run_level8_1_local.py --smoke-test
# Formal matched v0.1/v0.2 retention run:
python run_level8_1_local.py
```

Level 8.1 trains v0.1 and v0.2 with identical streams and compute schedules before evaluating the 1–1000 Chunk lifetime curve. Do not use its smoke output as a formal scientific result.

## Known risks

- The hierarchical modules are a minimal prototype, not a demonstrated retention improvement.
- Episodic replacement uses a configured hard eviction choice; its candidate values remain differentiable, but the selected index is not.
- Soft routing may collapse to one route without task pressure.
- Protected Slow slots can saturate or become stale.
- Fixed top-k Episodic retrieval can miss relevant details.
- v0.2 adds parameters and state; Level 8.1 must measure whether the lifetime gain, if any, justifies it.
- Natural language, large models, and external RAG are intentionally out of scope.
