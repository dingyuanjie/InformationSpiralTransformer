# Level 6.7: unified robust protocol

This is a new registered validation on five fresh model seeds: 101, 202, 303,
404, and 505. It does not reuse the Level 6.6 recovery seeds.

Unified changes derived from the recovery study:

- fixed-marker and random 2-chunk budgets increased to 2,500 steps;
- 4/8/16-chunk budgets increased to 1,500 steps;
- 16-chunk LR fixed at `1e-5`;
- withdrawal LR fixed at `5e-6`;
- zero-probe maintenance increased to 750 steps;
- EMA with decay 0.995 begins at withdrawal and is the preregistered primary
  final model; raw final weights are a paired control.

EMA uses no validation-based checkpoint selection. Raw and EMA weights are
evaluated on the same independent 400-example stream. The primary success gate
requires EMA query >=95% and minimum probe accuracy >=90%.

```powershell
python run_level6_7_local.py
```

Results are written under `experiments/level6_7/formal/` with per-stage
checkpoints and per-seed restart support.
