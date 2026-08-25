# Frozen Memory 0.4 — Generalization

Three fresh Fast-only persistence adapters train on non-repeating 1K examples. Independent validation selects checkpoints by normal-minus-zero-fast gap. Held-out evaluation is paired across Base, normal, zero-fast, and reset.

```powershell
python run_pretrained_frozen_memory_0_4.py --dry-run
python run_pretrained_frozen_memory_0_4.py --local-files-only
```
