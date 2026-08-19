# Level 7.5.3.1: unsuppressed recovery dynamics

Level 7.5.3 found no L2/L3 route switching. Six intervention endpoints remained
below the fresh 16-chunk formation gate despite recovering four-chunk training
performance. This experiment tests whether those deficits are temporary.

All twelve frozen C4 endpoints resume model, Probe, optimizer, and RNG state.
Every branch receives exactly 1,000 additional four-chunk steps at the unchanged
learning rate with no Memory mask or other intervention.

## Formal run

```powershell
python run_level7_5_3_1_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **5-8 hours**. Training
resumes at 100-step validation boundaries and panels resume by condition. After
interruption, run the same command without `--force`.

At recovery steps 0/100/300/600/1000, a shared N=1,024 five-condition screen
records intact behavior and L2/L3 retention. At step1000, a separate N=2,048
full sixteen-condition panel provides the registered final classification.

`complete_unsuppressed_route_recovery` requires all six initially unformed
branches to recover their original source-specific route and all six initially
formed branches to preserve it. The two groups are frozen from Level 7.5.3.

## Smoke test

```powershell
python run_level7_5_3_1_local.py --smoke-test --force
```

Smoke mode validates all source hashes, deterministic no-mask continuation,
and the trajectory-screen conditions. It is not scientific evidence.

## Outputs

Formal outputs are written under `experiments/level7_5_3_1/formal/`, including
`result.json`, `summary.json`, `progress.json`,
`unsuppressed_recovery_dynamics.png`, and resumable per-branch artifacts.

