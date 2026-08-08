# Level 6.5.2 analysis: cross-stream learning-rate stability

## Protocol

The same formed `hard400_seed313/stage3.pt` model and restored Adam state were
continued on five deterministic data streams. Each stream compared `5e-5`
against `1e-4` for 1,000 training and 500 maintenance steps. Probe loss was zero
and the probe was frozen.

## Preregistered strict pass

| LR | Strict passes |
| ---: | ---: |
| 5e-5 | 0/5 |
| 1e-4 | 1/5 |

The strict rule required a consecutive 95% gate, at least 95% exactly at step
1000, at least 95% on the 400-sample final evaluation, and at least 90% in every
one of the last five 40-sample intermediate evaluations. No `5e-5` stream met
all four conditions simultaneously. This registered result must not be
relabelled after observing the data.

## Accuracy comparison

| Data seed | Final 5e-5 | Final 1e-4 | Paired difference |
| ---: | ---: | ---: | ---: |
| 22026 | 96.00% | 93.25% | +2.75 pp |
| 42042 | 95.50% | 78.00% | +17.50 pp |
| 51234 | 98.25% | 63.00% | +35.25 pp |
| 70007 | 97.75% | 87.50% | +10.25 pp |
| 71313 | 91.25% | 98.75% | -7.50 pp |

- Mean final query: 95.75% for `5e-5`, 84.10% for `1e-4`.
- Mean paired improvement: +11.65 percentage points.
- `5e-5` won 4/5 paired streams.
- On the 400-sample final evaluation, `5e-5` reached 95% in 4/5 streams;
  `1e-4` did so in 1/5.
- Mean minimum over the last five intermediate evaluations was 89.5% versus
  74.5%.

The tail-min statistic is deliberately harsh and noisy: it selects the minimum
of five evaluations containing only 40 examples each. It is useful as a stress
test, but it should be read alongside the larger 400-example final evaluation.

## Interpretation

Lowering the 16-chunk learning rate substantially improves expected accuracy
and reduces catastrophic degradation across data streams. The benefit is not
universal: seed 71313 favored `1e-4`, and one `5e-5` stream ended at 91.25%.
Therefore `5e-5` is a better fixed setting, but it is not a 5/5 stability
solution under the strict protocol.

This experiment supports an optimization-stability explanation rather than a
hard 2,048-token memory-capacity limit. The same checkpoint frequently finishes
near 96--98% when updates are controlled, while the larger LR can collapse to
63%. However, stream-dependent reversals show that a single fixed LR is still
fragile.

## Recommended next step

Level 6.5.3 should test adaptive late-stage stabilization from the same
checkpoint: begin at `5e-5`, reduce to `1e-5` after the query first crosses the
gate or after validation deterioration, and preserve a best-checkpoint copy.
Compare this registered adaptive policy against fixed `5e-5` on the same five
streams. The goal is a 5/5 high-sample final result without relying on a lucky
fixed stopping step.

