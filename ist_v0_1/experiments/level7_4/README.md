# Level 7.4: seed1879 Memory-route formation trajectory

Level 7.4 freezes ten preexisting seed1879 checkpoints spanning 2/4/8/16-chunk
curriculum, two Probe-withdrawal phases, and four zero-Probe maintenance
milestones. Every checkpoint is evaluated on the same new 1,024-example
16-chunk panel under the same eleven Memory interventions.

The experiment asks when the L2-core/L3-support route first appears and whether
it remains stable after auxiliary-Probe withdrawal. Early curriculum
checkpoints that do not yet solve the 16-chunk task are classified as unformed,
not as evidence for a competing route.

No model, Probe, output head, or router is trained. No checkpoint is selected
or omitted after results are observed.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_4_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **25-45 minutes**. Each
checkpoint-condition result is saved immediately, so the run resumes after an
interruption. Do not add `--force` when resuming.

## Registered interpretation

The primary positive result requires the L2-core/L3-support signature at the
end of 16-chunk curriculum and at every later withdrawal/maintenance
checkpoint. Separate classifications cover late emergence, destabilization,
route migration, and an unresolved trajectory.

Thresholds describe broad causal route classes rather than repeating Level
7.3.1's high-precision 90% lower-bound test. Full point estimates and Wilson
intervals are still saved for every checkpoint and condition.

## Smoke test

```powershell
python run_level7_4_local.py --smoke-test --force
```

Smoke output is isolated under `experiments/level7_4/smoke/` and is not
scientific evidence.
