# Level 7.5: prospective cross-initialization formation dynamics

## Formal status

**Completed — `alternative_route_formation_observed`, integrity PASS.** The
registered seed1879 L2 two-stage hypothesis did not replicate, but all three
new initializations formed whole-Memory-causal 16-chunk behavior through the
same L3-dominant route. Endpoint queries were 97.17%, 97.27%, and 91.89%; full
Memory disruption collapsed them to chance while local accuracy remained
99.51%. See `formal/ANALYSIS.md` for the complete trajectory and scope limits.

Level 7.4.1 exactly replayed seed1879 and found an L2-core causal precursor
around C4 step600 before the fully qualified L2-core/L3-support route at
step1000. Level 7.5 prospectively tests that two-stage pattern on three new
initializations: `2203`, `2551`, and `2909`.

Every seed starts from scratch. Training stops after the C4 stage; there is no
C8/C16 curriculum, Probe withdrawal, seed replacement, repair, or post-result
extension. The C2 endpoint, C4 step1, and every 100-step C4 validation
milestone are saved before any formal result is observed.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_5_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **2.5-5 hours**. Runtime
depends on each seed's fixed/C2/C4 early-stop step and the number of registered
milestones. The run is restartable at completed training evaluations, causal
conditions, milestones, and seeds. Run the same command again after an
interruption; completed fixed/C2 endpoints, C4 evaluation milestones, causal
conditions, and seeds are reused. Do not add `--force` when resuming.

The causal panel uses the same new 1,024-example, 16-chunk dataset for all
seeds, milestones, and sixteen whole/layer/pair interventions. This detects an
L2 precursor without assuming that every new seed must use seed1879's route.

## Registered primary question

The cohort result is a strong two-stage replication only when at least two of
the three seeds show an L2 causal precursor after C2 step0 and later finish C4
with the full L2-core/L3-support route. Alternative L1/L3/distributed routes
remain valid secondary findings but cannot be relabeled as the registered L2
replication.

## Smoke test

```powershell
python run_level7_5_local.py --smoke-test --force
```

Smoke mode uses a tiny isolated seed, three miniature C4 steps, four chunks,
and 32 samples. It tests milestone saving, all sixteen interventions, frozen
fingerprints, diagnosis, and resume artifacts. It is not scientific evidence.

## Outputs

Formal artifacts are written under `experiments/level7_5/formal/`, including
per-seed training and causal checkpoints, `result.json`, `summary.json`,
`progress.json`, `prospective_formation_dynamics.png`, and the frozen protocol.
