# Base Smoke 0.2 — Fixed-set Overfit

Overfit 32 fixed 1K examples. Pass only if normal reaches 95% and exceeds both zero-memory and reset-memory by 50 points in two consecutive checks. A pass proves learnability/causal use, not generalization.

```powershell
python run_pretrained_base_smoke_0_2.py --dry-run
python run_pretrained_base_smoke_0_2.py --local-files-only
```
