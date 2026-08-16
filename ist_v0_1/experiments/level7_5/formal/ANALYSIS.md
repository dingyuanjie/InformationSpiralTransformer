# Level 7.5 prospective cross-initialization formation dynamics analysis

## Registered result

The formal run completed with integrity **PASS**. The registered cohort
classification is:

`alternative_route_formation_observed`

The preregistered seed1879-specific hypothesis did **not** replicate. None of
the three untouched initializations showed the registered L2 causal precursor
or finished C4 with the L2-core/L3-support route. This is not a failure of
persistent Memory formation: all three seeds formed high-accuracy,
whole-Memory-causal 16-chunk behavior through the same alternative L3-core
route.

## Integrity

- Formal seeds were exactly `2203`, `2551`, and `2909`.
- All three were new initializations; no old model or checkpoint was reused.
- All fixed, C2, and C4 budgets and early-stop rules remained locked.
- Training stopped after C4; no C8, C16, Probe withdrawal, repair, or rescue
  was run.
- All 18 expected frozen milestones were evaluated.
- Every milestone used all 16 registered conditions at fixed N=1,024 on the
  same new dataset seed `7500000`.
- All frozen-model fingerprints were unchanged during diagnostics.
- No seed was replaced and seed909 remained closed.

## Training and endpoint behavior

| Seed | Fixed stop | C2 stop | C4 stop | Fresh 16-chunk endpoint | 95% Wilson CI | Minimum local | Endpoint route |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2203 | 1500 | 1000 | 500 | 97.17% | 95.96%-98.02% | 99.51% | L3 core |
| 2551 | 1300 | 800 | 300 | 97.27% | 96.08%-98.10% | 99.51% | L3 core |
| 2909 | 1400 | 800 | 400 | 91.89% | 90.06%-93.41% | 99.51% | L3 core |

All seeds passed their fixed, C2, and C4 training gates without budget
extension. The final in-range C4 validation queries were 97.50%, 98.75%, and
100.00%, respectively. Fresh 16-chunk behavior was therefore already strong
at the C4 stop, although seed2909 retained a visible in-range-to-extrapolation
gap.

## Whole-Memory causal replication

| Seed | Intact | Reset all | Zero all | Roll all | Local |
|---:|---:|---:|---:|---:|---:|
| 2203 | 97.17% | 6.45% | 7.13% | 4.39% | 99.51% |
| 2551 | 97.27% | 6.35% | 6.35% | 4.59% | 99.51% |
| 2909 | 91.89% | 6.35% | 6.35% | 4.69% | 99.51% |

Resetting, zeroing, or cross-sample rolling the complete persistent Memory
collapsed every endpoint to approximately the 6.25% chance level while local
accuracy remained 99.51%. Thus Level 7.5 adds three prospective independent
replications of complete persistent-Memory causality.

## Endpoint layer mechanism

| Seed | Zero L1 | Zero L2 | Zero L3 | Roll L3 | Keep L1 | Keep L2 | Keep L3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2203 | 97.85% | 97.27% | 6.64% | 4.49% | 6.45% | 7.03% | 98.14% |
| 2551 | 97.46% | 97.17% | 5.86% | 4.49% | 5.76% | 6.35% | 97.66% |
| 2909 | 91.41% | 91.41% | 6.35% | 4.69% | 6.35% | 6.35% | 91.50% |

All three endpoint signatures were exactly identical:

- necessary layer: L3 only;
- batch-roll-sensitive layer: L3 only;
- sufficient single layer: L3 only;
- sufficient pairs: L1+L3 and L2+L3;
- L1 and L2 alone: near chance.

This is a clean L3-dominant mechanism. Removing L1 or L2 leaves intact
performance, whereas removing or misassigning L3 destroys the behavior. L3
alone reconstructs the intact endpoint.

## Formation trajectory

| Seed | C2 endpoint query | Last unformed milestone | First L3-core milestone | Endpoint |
|---:|---:|---:|---:|---:|
| 2203 | 24.41% | step400: 87.21% | step500: 97.17% | step500 |
| 2551 | 34.96% | step200: 89.84% | step300: 97.27% | step300 |
| 2909 | 28.12% | step200: 41.41% | step300: 93.26% | step400: 91.89% |

The L3 route has a weak precursor before formal behavior formation. Already at
the C2 endpoints, zero-L3 was near chance and keep-L3 retained 28%-37% query
accuracy, while intact behavior was only 24%-35%. During C4 training this
L3-specific scaffold was amplified until it crossed the registered 90%
behavior and sufficiency thresholds.

The preregistered **L2** two-stage sequence therefore failed, but the broader
mechanistic pattern survived in a corrected form: weak layer-selective state
can precede full 16-chunk behavior, and the selected layer was consistently L3
in all three new initializations.

## Updated interpretation

Level 7.3 previously found L3 dominance in seeds 606, 808, and 1001, with
seed1879 using an exceptional L2-core/L3-support route. Level 7.5 adds three
new L3-dominant initializations. Across the currently frozen formed-checkpoint
evidence set, six models use L3 dominance and one uses the seed1879 L2 route.
This is an evidence trend, not a population-frequency estimate, because the
checkpoints came from related but non-identical training stages and selection
protocols.

The supported conclusion is now narrower and stronger:

> Persistent Memory causality and four-to-sixteen-chunk extrapolation replicate
> across new initializations, while the seed1879 L2 route and its exact
> two-stage formation dynamics do not. Under the current fresh C4 protocol,
> L3 dominance is the reproducible default route.

## Stop boundary and next test

Do not replace, extend, or selectively continue any Level 7.5 seed. The most
informative next experiment is an endpoint-qualified deterministic replay of
the fixed-to-C2 interval for the three L3 seeds and seed1879. That comparison
can localize when the L3 precursor appears and why seed1879 fails to select it,
without inserting post-hoc checkpoints into the completed Level 7.5 training
run.
