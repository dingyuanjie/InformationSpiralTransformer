# Level 6.8: behavior-first unified protocol

This registered experiment uses five fresh model seeds: 606, 707, 808, 909,
and 1001.

Changes from Level 6.7 are limited to the diagnosed factors:

- 8-chunk LR is reduced from `2.5e-4` to `5e-5`;
- 16-chunk LR remains `1e-5`;
- extended Level 6.7 budgets are retained;
- withdrawal LR remains `5e-6` with 750 zero-probe steps;
- EMA is removed because it did not improve Level 6.7;
- two consecutive query evaluations >=95% are the primary curriculum gate;
- final query >=95% is the primary end-to-end success gate;
- probe minimum >=90% is reported as a secondary diagnostic and does not veto
  behavioral success.

The fixed-marker scaffold still requires its probe criteria because direct
memory decoding is the purpose of that initialization stage.

```powershell
python run_level6_8_local.py
```

Results are written under `experiments/level6_8/formal/` with per-stage and
per-seed restart support.
