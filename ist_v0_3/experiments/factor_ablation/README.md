# v0.3 Milestone 2.2.2: orthogonal factor ablation

The locked Reader is evaluated at 8 chunks under one changed factor at a time:
answer vocabulary, templates, target position and distractor facts. The full
strict condition is included. A separate 32-chunk panel varies Memory capacity
across 64/128/256 slots without changing learned weights.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_factor_ablation.py --dry-run
python run_v0_3_factor_ablation.py --local-files-only
```

Results and full logs are written to `results.json` and `run.log`.
