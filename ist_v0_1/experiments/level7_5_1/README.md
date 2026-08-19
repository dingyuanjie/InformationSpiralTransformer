# Level 7.5.1: fixed-to-C2 route-bifurcation replay

## Formal status

The formal run completed with integrity **PASS** and classification
`default_L3_precursor_divergence_confirmed`. The weak L3 precursor first
appeared at C2 step1000 for seed2203 and step700 for seeds2551/2909, then
persisted to each endpoint. Seed1879 never entered the registered L3 route
through its longer step2300 C2 replay. A clearly labeled post-hoc mirrored
analysis found a transient weak L2 scaffold only in seed1879 at steps1400 and
1600; this now requires independent confirmation. See
`formal/ANALYSIS.md` for the complete evidence and scope limits.

Level 7.5 found the same L3-dominant endpoint in three untouched
initializations, while seed1879 remained the only frozen L2-core/L3-support
solution. At every Level 7.5 C2 endpoint, weak L3 selectivity was already
present. Level 7.5.1 moves one stage earlier and asks when that default L3
precursor first appears and whether seed1879 ever enters it.

The script restores each model's fixed-stage model, Probe, optimizer, and RNG
state; performs the original `seed + 20000` reset; and replays the unchanged C2
stage. A seed's frozen causal trajectory is opened only if its replayed C2
endpoint exactly matches the original model, Probe, optimizer, CPU/CUDA RNG,
validation history, consecutive-pass state, and stop step.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_5_1_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **4-7 hours**. There are
57 registered milestones across four trajectories, and every qualified
milestone receives all sixteen N=1,024 16-chunk interventions. The run resumes
at replay validation milestones, causal conditions, milestones, and seeds.
After interruption, execute the same command without `--force`.

## Registered primary outcome

`default_L3_precursor_divergence_confirmed` requires all four exact replay
gates, prospective L3-precursor formation in seeds2203/2551/2909, and no such
precursor anywhere in seed1879's fixed-to-C2 trajectory. A mismatch closes the
affected seed rather than permitting approximate replay interpretation.

## Smoke test

```powershell
python run_level7_5_1_local.py --smoke-test --force
```

Smoke mode validates the four real source hashes, miniature exact replay,
sixteen-condition frozen evaluation, profile computation, and cleanup. It is
not scientific evidence.

## Outputs

Formal outputs are written under `experiments/level7_5_1/formal/`, including
per-seed replay gates and causal trajectories, `result.json`, `summary.json`,
`progress.json`, `fixed_to_C2_route_bifurcation.png`, and the frozen protocol.
