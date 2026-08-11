# Level 6.18.1: rollback-and-bridge recovery

Level 6.18 showed that another 3,000 updates at the failed context length did
not stably recover seeds 707 or 909. Level 6.18.1 tests a different hypothesis:
the optimizer must return to the last proven curriculum checkpoint and cross
the length transition through an intermediate context while rehearsing shorter
contexts.

## Frozen roles and schedule

- seed 707 is the calibration run and is evaluated first;
- seed 909 is a locked transfer run and is opened only if seed 707 passes;
- seed 707 rolls back to the passed 4-chunk checkpoint and follows
  `4 -> 6 -> 8 -> 12 -> 16`;
- seed 909 rolls back to the passed 8-chunk checkpoint and follows
  `8 -> 12 -> 16`.

For every `base -> target` transition:

- bridge updates repeat `[base, midpoint, midpoint, midpoint]`;
- target updates repeat
  `[base, midpoint, target, target, target, target]`.

The bridge phase has at most 1,500 updates at learning rate `2e-5`. The target
phase has at most 3,000 updates at learning rate `1e-5`. The old failed
checkpoints are not loaded or overwritten.

## Stability gate

Every 100 updates, an 80-example fixed screen is evaluated. Screens at or above
90% activate two disjoint fixed 200-example confirmation panels. A checkpoint
is confirmed when:

- pooled query accuracy is at least 95%; and
- the worse panel is at least 93%.

Two successive evaluated checkpoints must be confirmed. Probe accuracy is
recorded but is diagnostic only. The script stores separate `latest`, `best`,
and formally accepted `stable` checkpoints, so a transient peak is never
silently promoted to success.

## Post-formation gates

After reaching 16 chunks, the run performs the Level 6.8 probe-withdrawal
schedule (`0.2 -> 0.1 -> 0.0`). Final behavioral accuracy must remain at least
95%. It then evaluates intact, reset, zero, and batch-rolled memory on 400 fixed
examples per condition. The causal gate requires:

- intact query accuracy at least 90%;
- every disrupted-memory accuracy at most 20%;
- local accuracy at least 90%.

The complete frozen protocol is written to `formal/preregistration.json` before
GPU optimization starts.

## Run

From the repository root:

```powershell
python run_level6_18_1_local.py
```

The run is resumable at every 100-step evaluation checkpoint. Do not add
`--force` when resuming. On an RTX 5060 Laptop GPU, budget roughly 1-2 hours if
neither seed reaches its gates early.

Results are written under `experiments/level6_18_1/formal/`. The formal command
must use the defaults. Shortened steps, changed thresholds, or a different
output directory are smoke tests or exploratory runs and must not be reported
as the preregistered Level 6.18.1 result.

## Interpretation

- If seed 707 fails, seed 909 remains unopened and the rollback protocol is a
  calibration failure.
- If seed 707 passes but seed 909 fails, rollback-and-bridge recovery did not
  transfer across initialization.
- If formation passes but withdrawal fails, the recovered circuit still
  depends on direct probe supervision.
- If withdrawal passes but the causal gate fails, high query accuracy is not
  sufficient evidence of restored persistent memory.
- Only two complete formation, withdrawal, and causal passes support recovery
  transfer. Seed 909 is locked transfer rather than a new pristine held-out
  initialization because its Level 6.18 learning curve has already been
  observed.

