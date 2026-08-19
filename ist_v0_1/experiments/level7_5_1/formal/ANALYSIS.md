# Level 7.5.1 formal analysis

## Outcome

The formal run completed with integrity **PASS** and the registered
classification:

`default_L3_precursor_divergence_confirmed`

All three default L3 trajectories formed the preregistered weak L3 precursor
during the fixed-to-C2 interval. The exceptional seed1879 trajectory never
met that L3 rule anywhere in its longer C2 replay. The route divergence is
therefore already present during C2; it is not first created during the later
C4 curriculum.

This run took 13,315.5 seconds (3 h 41 min 56 s) and completed all 57
registered milestones.

## Integrity and exact replay

The causal trajectory of a seed was opened only after its replayed C2 endpoint
exactly matched the original endpoint. All four seeds passed every component:

| Seed | Model | Probe | Optimizer | CPU/CUDA RNG | Validation history | Stop state | Milestones |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2203 | exact | exact | exact | exact | exact | exact | 12/12 |
| 2551 | exact | exact | exact | exact | exact | exact | 10/10 |
| 2909 | exact | exact | exact | exact | exact | exact | 10/10 |
| 1879 | exact | exact | exact | exact | exact | exact | 25/25 |

All source hashes were validated. Every qualified milestone used the same new
N=1,024, 16-chunk dataset seed (`7510000`) and all 16 registered causal
conditions. No checkpoint was trained beyond the original C2 stopping point,
and no failed or approximate replay entered the result.

## Registered weak-L3 result

The weak precursor rule was frozen before the run. In brief, it required
nontrivial intact behavior, at least 90% local performance, near-complete
failure after destroying or rolling L3, survival when only L3 was retained,
failure when only L1 or L2 was retained, and a minimum L3 selectivity margin.
It was intentionally weaker than the later strict L3-core endpoint rule.

| Seed | Later route | C2 stop | First weak L3 | Intact at first | `keep_L3` at first | L3 margin | C2 endpoint intact |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2203 | L3-dominant | 1000 | step 1000 | 24.12% | 33.01% | +26.95 pp | 24.12% |
| 2551 | L3-dominant | 800 | step 700 | 22.56% | 23.34% | +17.77 pp | 34.18% |
| 2909 | L3-dominant | 800 | step 700 | 24.80% | 24.80% | +19.24 pp | 26.66% |
| 1879 | L2-core/L3-support | 2300 | never | — | — | — | 17.77% |

The result is not a one-point endpoint artifact:

- seed2551 and seed2909 first selected L3 at step700 and retained that choice
  through their C2 endpoints;
- seed2203 narrowly missed the frozen floor at step900 (`zero_L1=19.14%`) and
  crossed the complete rule at step1000;
- each default seed made one unformed-to-L3 transition and showed no route
  regression;
- seed1879 made no L3 transition across 25 inspected milestones through
  step2300. Its endpoint L3 margin was only +0.78 percentage points.

None of the four C2 trajectories yet met the later strict L3-core criterion.
That is consistent with Level 7.5.1 localizing a weak developmental precursor,
not claiming that full long-context behavior was complete at C2.

## Exploratory secondary observation: a weak L2 scaffold in seed1879

After seeing the registered result, the same thresholds were mirrored from L3
to L2 as a post-hoc diagnostic. This analysis was **not preregistered** and is
therefore hypothesis-generating rather than part of the primary
classification.

The mirrored rule selected only seed1879:

| Seed | Step | Intact | `keep_L2` | L2 selectivity margin |
|---:|---:|---:|---:|---:|
| 1879 | 1400 | 25.68% | 24.71% | +18.46 pp |
| 1879 | 1600 | 33.59% | 30.08% | +21.58 pp |

It selected no milestone from seeds2203, 2551, or 2909. Seed1879's intact
score peaked at 37.21% at step1700 while its L2-retention response was elevated,
then weakened before the C2 endpoint. This suggests a transient L2 scaffold
rather than an absence of route formation. Together with Level 7.4.1—where
seed1879's L2 precursor reappeared in C4 and the full L2-core/L3-support route
qualified at step1000—the most economical developmental account is:

1. fixed-stage behavior is causally unformed;
2. C2 begins route selection: the three default seeds select weak L3 around
   step700–1000, while seed1879 transiently selects weak L2 around
   step1400–1600;
3. C4 amplifies the selected scaffold into a stable route-specific circuit.

The L2 claim must be repeated on an independently frozen panel before it can
be promoted from exploratory evidence.

## What this establishes

- Exact optimization replay reproduces each original C2 endpoint, so the
  intermediate trajectories are auditable rather than approximate reruns.
- Three independent default initializations converge on the same early L3
  selection despite different first-crossing steps.
- The exceptional seed1879 does not first follow L3 and later migrate to L2.
  Its divergence is already visible within C2.
- The combined Level 7.4.1, 7.5, and 7.5.1 evidence supports
  initialization-dependent route choice followed by curriculum-dependent
  amplification.

The experiment does not estimate route frequencies in a population, prove
that initialization alone determines route choice, or establish the post-hoc
L2 precursor as a registered result. There are four trajectories, one
synthetic task family, and one architecture/configuration.

## Next experiment

Level 7.5.2 should preregister an **independent weak-L2 precursor confirmation**.
Freeze the existing seed1879 step1200–1800 milestones before evaluation, use a
new shared causal dataset, apply the mirrored L2 rule without modification,
and include matched milestones from the three default L3 seeds as negative
controls. This is the shortest valid test of whether the apparent C2 L2
scaffold replicates out of sample.

Only after that confirmation should Level 7.5.3 perform a training-time
counterfactual intervention near route commitment (suppress or preserve the
candidate L2/L3 state over a fixed window) to distinguish a causally directing
precursor from a correlated early readout.

## Artifacts

- `result.json`: complete raw formal result and integrity records
- `summary.json`: compact seed and cohort summaries
- `progress.json`: resume/completion state
- `fixed_to_C2_route_bifurcation.png`: registered trajectory visualization
- `seed*/`: per-seed exact replay gates, milestones, and causal panels

