# Base Smoke 0.3 — Persistence Only

The adapter delta is `Memory(hidden, historical_state) - Memory(hidden, reset_state)`. Query-local transformations cancel by construction. No-history and reset logits must exactly equal Base before training.

```powershell
python run_pretrained_base_smoke_0_3.py --dry-run
python run_pretrained_base_smoke_0_3.py --local-files-only
```
