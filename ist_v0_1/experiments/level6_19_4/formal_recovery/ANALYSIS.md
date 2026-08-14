# Level 6.19.4 formal analysis

## Decision

- Registered classification: `oracle_not_compiled_into_label_free_router`.
- The smallest sets reaching either 80% or 90% recovery contain all eight
  heads: `[0,1,2,3,4,5,6,7]`.
- The exact signed target is infeasible in the valid attention simplex for all
  241 primary examples after the convex solver passes both convergence gates.
- The fixed signed router recovers 23.31% of the full signed-Oracle margin
  gain, below the registered 25% threshold.
- Five of six signed-specificity contrasts pass. The signed router does not
  outperform the equal-parameter matched residual router.
- The trunk remains frozen. Seed909 and protected tests remain closed.

## Numerical recovery and integrity

The first completed run in `../formal/` used 2,048 simplex iterations and was
retained as an integrity failure. The isolated recovery increased only the
deterministic convex-solver budget to 8,192 iterations. The scientific target,
tolerances, data, splits, router checkpoints, and registered decision rule were
unchanged.

The recovery passes every integrity gate:

- exact source downstream reconstruction: maximum absolute error `0`;
- selected-Oracle L2 error: `4.77e-7`;
- full-Oracle L2 error: `5.01e-6` (registered maximum `1e-5`);
- simplex convergence: passed;
- frozen model/probe fingerprints: unchanged;
- frozen parameters: still frozen;
- router parameter counts: equal at 8,930 each;
- split seeds: unique except the registered parent replay;
- formal diagnostic: isolated from training and selection;
- failed Level 6.18.9 candidate: excluded;
- seed909 and protected tests: unopened;
- optimizer/model search: closed.

The calibration, training, and checkpoint files have identical SHA-256 hashes
between the original and recovery runs. Labels, groups, and every non-simplex
diagnostic condition are also exactly identical. This establishes that the
recovery repaired numerical convergence rather than changing the experiment.

## All-subset causal tomography

The calibration replay contains 4,096 examples, 313 source errors, and 257
Memory-decodable source errors. The full signed Oracle has mean deployed-margin
gain `0.160657`. The best subset at each cardinality is:

| Heads | Best subset | Full-signed recovery |
|---:|---|---:|
| 1 | `[7]` | 19.38% |
| 2 | `[4,7]` | 28.53% |
| 3 | `[4,6,7]` | 36.25% |
| 4 | `[2,4,6,7]` | 43.83% |
| 5 | `[0,4,5,6,7]` | 53.10% |
| 6 | `[0,2,3,5,6,7]` | 64.12% |
| 7 | `[0,2,3,4,5,6,7]` | 78.92% |
| 8 | `[0,1,2,3,4,5,6,7]` | 99.14% |

No one-to-seven-head subset reaches 80%; therefore the registered 80% and 90%
sets are both the complete eight-head set. The result supports a distributed
mechanism with no sufficient small head core under this equal-L2 deployed
intervention. These subsets are label-aware causal Oracles, not deployable
readers.

## Exact target-simplex audit

On the fresh 241-example primary population:

| Quantity | Result |
|---|---:|
| feasible fraction at relative residual <= 0.01 | 0/241 (0%) |
| mean relative residual | 0.6302 |
| median relative residual | 0.6418 |
| maximum relative residual | 0.8556 |
| maximum relative two-start delta gap | 0.004006 (gate 0.01) |
| maximum projected-gradient mapping | 1.84e-6 (gate 1e-5) |

The convergence gates now pass with margin, yet every exact projection remains
far from the signed target. Thus the registered signed target is outside the
product of valid per-head attention simplices. This result concerns exact
target representability; it does not claim that every possible equal-dose
simplex direction is ineffective.

## Router training

All three readers have 8,930 trainable parameters and use the same observable
inputs, split, selected head mask, optimizer protocol, and global dose ceiling.
Training uses labels and Oracle distillation; inference uses neither the target
label nor a rival class.

| Router | Best epoch | Validation task loss | Distillation loss | Accuracy | Mean gate |
|---|---:|---:|---:|---:|---:|
| signed value-basis | 29 | 0.28315 | 0.60777 | 91.60% | 0.692 |
| non-negative attention | 30 | 0.28667 | 0.85204 | 91.41% | 0.850 |
| matched residual | 30 | 0.28320 | 0.58778 | 91.60% | 0.688 |

The non-negative reader has the weakest distillation fit. The residual reader
slightly outperforms the signed reader on distillation while matching its task
accuracy, which anticipates the formal specificity failure.

## Fresh diagnostic panel

The one-shot formal panel contains 4,096 samples. Source accuracy is 92.9688%,
with 288 source errors and 241 Memory-decodable source errors (the primary
population).

| Condition | Full accuracy | Primary margin gain | Primary correction | Primary context L2 | Primary gate |
|---|---:|---:|---:|---:|---:|
| source | 92.9688% | 0 | 0% | 0 | 0 |
| signed router | 93.0664% | +0.03804 | 1.66% | 0.7446 | 0.862 |
| non-negative router | 92.9688% | +0.00156 | 0% | 0.0289 | 0.901 |
| matched residual router | 93.0664% | +0.04162 | 1.66% | 0.7502 | 0.868 |
| shuffled-memory signed | 92.9932% | +0.01644 | 0.41% | 0.6962 | 0.806 |
| rolled-delta signed | 92.9443% | -0.00044 | 0% | 0.6057 | 0.701 |
| head-permuted signed | 92.9443% | +0.00664 | 0.41% | 0.7446 | 0.862 |
| selected-subset Oracle | 93.3105% | +0.16320 | 4.98% | 0.8674 | n/a |
| full signed Oracle | 93.3105% | +0.16320 | 4.98% | 0.8674 | n/a |

Because the selected subset contains every head, the selected and full Oracles
are identical on the fresh panel. Across the full panel, the signed router
corrects four examples and introduces no regressions: accuracy changes by
`+0.0977` percentage points, with bootstrap 95% CI `[+0.0244,+0.1953]`
percentage points. The registered noninferiority gate therefore passes. The
four-discordance McNemar p-value is 0.125, so this is not promoted as a
standalone accuracy-superiority claim.

## Registered specificity

All effects below are paired fixed-margin contrasts on the 241 primary
examples. P-values are Holm-corrected over the six registered comparisons.

| Contrast: signed minus condition | Estimate | 95% CI | Holm p | Pass |
|---|---:|---:|---:|---|
| source | +0.03804 | [0.03250, 0.04339] | 0.000600 | yes |
| non-negative router | +0.03648 | [0.03128, 0.04178] | 0.000600 | yes |
| matched residual router | -0.00358 | [-0.00794, 0.00086] | 0.118 | no |
| shuffled-memory signed | +0.02159 | [0.01639, 0.02686] | 0.000600 | yes |
| rolled-delta signed | +0.03847 | [0.03135, 0.04549] | 0.000600 | yes |
| head-permuted signed | +0.03140 | [0.02601, 0.03686] | 0.000600 | yes |

The signed router's full-Oracle recovery is `0.03804 / 0.16320 = 23.31%`,
missing the registered 25% gate by 1.69 percentage points. It also fails the
matched-residual specificity contrast. On 288 confidence-matched source-correct
examples it preserves 100% accuracy while increasing mean margin by 0.05396;
this suggests that the learned gate is not specifically identifying the error
population.

## Scientific conclusion

Level 6.19.4 separates three facts:

1. The label-aware correction is genuinely distributed across all eight read
   heads; no seven-head core reaches even 80% recovery.
2. Its exact signed target is geometrically unavailable to ordinary valid
   non-negative attention, even after an independently converged convex audit.
3. A fixed label-free signed router learns a real memory-, alignment-, and
   head-specific effect, but it does not compile enough of the Oracle and is
   not specific to the projected memory-value basis: an unrestricted matched
   residual reader is at least as effective.

The result is therefore a successful mechanism localization but a failed
deployment compilation. It does not justify opening seed909, protected tests,
or model/optimizer search.

## Next experiment

Proceed to the frozen Level 6.19.5 observability-supervision diagnosis defined
in `../NEXT_STAGE.md`. Its purpose is to determine whether the 23.31% ceiling
comes from failure to infer when/how much to intervene, failure to infer the
signed direction from label-free observables, or the signed value-basis
parameterization itself.
