# Level 7.4.1: deterministic dense C2-to-C4 replay

Level 7.4 localized seed1879's transition from unformed 16-chunk behavior at
the saved C2 endpoint to a stable L2-core/L3-support route at C4. Level 7.4.1
restores the original C2 model, Probe, AdamW optimizer, CPU RNG, and CUDA RNG,
then replays the unchanged C4 curriculum stage.

The replay saves model milestones at step 1 and every 100 steps. It reproduces
the original validation schedule and early stopping because those validation
calls consume RNG and are part of the training trajectory.

## Qualification before causal analysis

The fresh causal panel remains closed unless the replay endpoint exactly
matches the original C4 state in all of these components:

- model tensors;
- Probe tensors;
- optimizer state;
- CPU and CUDA RNG states;
- stage-2 validation history;
- stop step and consecutive-pass state.

A newly serialized `.pt` file is not expected to have the same file hash, so
the gate uses canonical component fingerprints and byte-exact RNG comparison.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_4_1_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **35-55 minutes**. The
exact replay uses one rolling full-state resume file plus compact model
milestones. Frozen causal evaluation resumes per milestone and condition. Do
not add `--force` when resuming.

## Causal localization

After qualification, twelve milestones (C2 step0 plus step 1 and every 100
steps through 1000) are evaluated on the same new N=1,024 16-chunk panel under
the eleven Level 7.4 interventions. The output identifies the last non-target
and first stable L2-core/L3-support milestone, narrowing formation to at most a
100-step interval.

## Smoke test

```powershell
python run_level7_4_1_local.py --smoke-test --force
```

Smoke mode performs an isolated miniature reference/replay equality test. It
does not replay the formal C2-to-C4 interval and is not scientific evidence.
