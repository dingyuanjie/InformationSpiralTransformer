# Frozen Memory 0.4.1 — Nonzero Gate Warmup

Each seed starts with fresh Fast Memory and persistence scale -0.01. The gate is fixed for 500 steps, then trained at LR 2e-6. No fixed-set Memory weights are loaded.

```powershell
python run_pretrained_frozen_memory_0_4_1.py --dry-run
python run_pretrained_frozen_memory_0_4_1.py --local-files-only
```
