# Level 7.1: independent-initialization replication

## Formal status

**Completed — `formation_replication_failed`.** Both new initializations passed
all curriculum stages but failed the separate final 16-chunk formation gate
after Probe withdrawal: seed1217 reached 93.75% and seed1429 reached 90.625%
against the registered 95% threshold. The conditional causal and fresh read-gap
panels were therefore not opened. See `formal/ANALYSIS.md` and
`STOP_BOUNDARY.md`. No seed may be extended or replaced.

Level 7.1 trains two completely new IST initializations (`1217`, `1429`) from
scratch. It independently retests persistent 16-chunk behavior, all-Memory and
final-layer causal interventions, and the fresh Memory-versus-query-hidden
decodability gap. It does not use seed909, an old model checkpoint, a Level
6.19 evaluation split, or a router repair.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_1_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **45–80 minutes**. The
exact time depends on how early each curriculum stage reaches its two-
consecutive-evaluation gate.

The run is restartable. If it is interrupted, run the same command again. It
restores the latest completed fixed, curriculum, or withdrawal checkpoint. Do
not add `--force` when resuming.

## Implementation smoke test

```powershell
python run_level7_1_local.py --smoke-test --force
```

Smoke output is isolated under `experiments/level7_1/smoke/`, uses seed `17`,
and is never scientific evidence.

## Registered outcomes

- `strong_independent_replication`: both fresh initializations form and pass
  every causal gate;
- `conditional_independent_replication`: exactly one forms and passes, while
  the other stops at formation;
- `causal_replication_failed`: a formed model fails the causal gate;
- `formation_replication_failed`: neither model forms.

The secondary read-gap result is reported independently and cannot upgrade or
downgrade the primary result. A failed seed may not receive extra steps, a new
learning rate, output-head rescue, or replacement by a third seed.

## Formal artifacts

The formal directory will contain per-seed restart checkpoints and results,
plus `preregistration.json`, `result.json`, `summary.json`,
`independent_replication.png`, `progress.json`, and a completed `ANALYSIS.md`.
