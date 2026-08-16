# Level 7.2: validation-selected retention checkpoint

## Formal status

**Completed — `causal_gate_failed`.** Seed1601 stopped at the 4-chunk
curriculum. Seed1879 selected the registered 750-step checkpoint and passed the
one-time 4,096-example protected panel at 97.0947%. Complete Memory disruption
then reduced query accuracy to approximately 6.2%, but final-layer-only
necessity/sufficiency did not replicate: zero-L3 retained 91.60% and keep-L3
reached only 11.91%. See `formal/ANALYSIS.md` and `STOP_BOUNDARY.md`.

Level 7.1 showed that both new models crossed every curriculum gate but failed
after the complete zero-Probe maintenance tail. Level 7.2 tests one new
hypothesis with untouched seeds 1601 and 1879: the fixed final checkpoint may
be an unstable endpoint, while a large validation panel can select a stable
checkpoint from the same unchanged training trajectory.

No optimizer setting, loss, learning rate, training budget, or withdrawal
weight changes. Candidate zero-Probe steps are frozen at 300, 450, 600, and
750. A 1,024-example validation panel selects at most one eligible checkpoint;
only that checkpoint may open the one-time 4,096-example protected test.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_2_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **50–90 minutes**. Re-run
the identical command after interruption; completed curriculum, withdrawal,
and candidate checkpoints are restartable. Do not use `--force` when resuming.

If the selected checkpoint passes protected behavior, the script automatically
runs the preregistered seven-condition causal audit on a different split. If
selection or protected behavior fails, causality remains unopened.

## Smoke test

```powershell
python run_level7_2_local.py --smoke-test --force
```

Smoke output uses seed 23 under `experiments/level7_2/smoke/` and is not
scientific evidence.

## Boundary

Level 7.2 cannot rescue Level 7.1 seeds, add candidate steps, select on the
protected test, change training, add a third seed, repair an output head or
router, or open seed909.
