# Level 6.6 post-hoc targeted recovery

This experiment is deliberately separate from the registered Level 6.6 result
of 2/5. It does not revise that success rate.

Three diagnosed failures are tested from their original checkpoints:

- `seed7_budget`: extend the 4-chunk transition by 300 steps, then resume the
  remaining curriculum and standard withdrawal.
- `seed42_lr`: restart from the passed 8-chunk checkpoint and use `1e-5` for the
  16-chunk transition before standard withdrawal.
- `seed2026_withdrawal`: restart from the passed 16-chunk checkpoint and use a
  slower 0.3/0.2/0.1/0.05/0 probe schedule at learning rate `5e-6`.

Each recovery is successful only if the final 400-example query is at least 95%
and minimum per-chunk probe accuracy is at least 90%.

```powershell
python run_level6_6_recovery_local.py
```

Results are stored under `experiments/level6_6_recovery/formal/`.
