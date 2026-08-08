# Level 6.5.3 analysis: adaptive late-stage stabilization

## Result

The registered policy began at `5e-5` and permanently reduced to `1e-5` after
a 95% validation crossing, a 10-point deterioration signal, or step 600. Five
data streams were evaluated from the same formed checkpoint.

| Seed | Switch | Reason | Last test | Selected-best test |
| ---: | ---: | --- | ---: | ---: |
| 22026 | 300 | gate crossed | 98.25% | 97.00% |
| 42042 | 200 | gate crossed | 97.50% | 97.00% |
| 51234 | 300 | deterioration | 87.75% | 86.00% |
| 70007 | 100 | gate crossed | 97.75% | 95.50% |
| 71313 | 200 | gate crossed | 97.25% | 96.50% |

- Last-state high-sample success: 4/5.
- Validation-selected checkpoint success: 4/5.
- Mean last-state query: 95.70%; worst: 87.75%.
- Mean selected-checkpoint query: 94.40%; worst: 86.00%.
- Mean maintenance tail minimum: 93.00%.

## Paired comparison with fixed 5e-5

| Seed | Adaptive last | Fixed 5e-5 | Difference |
| ---: | ---: | ---: | ---: |
| 22026 | 98.25% | 96.00% | +2.25 pp |
| 42042 | 97.50% | 95.50% | +2.00 pp |
| 51234 | 87.75% | 98.25% | -10.50 pp |
| 70007 | 97.75% | 97.75% | 0.00 pp |
| 71313 | 97.25% | 91.25% | +6.00 pp |

The mean paired difference is -0.05 percentage points: effectively no change.
Both policies achieve 4/5 high-sample final successes. Adaptive decay improves
the average minimum of small maintenance evaluations (93.0% versus 89.5%), but
it does not improve the final success count or mean accuracy.

## Why best-checkpoint selection did not help

The intermediate validation sets contain only 40 examples. Selecting the
single highest observation favors sampling noise. The chosen checkpoints had
lower independent-test accuracy than the last states on every stream. This is
a useful negative result: naive max-validation checkpointing is not justified
at this evaluation size.

Seed 51234 triggered decay after the validation sequence 77.5%, 92.5%, 82.5%.
It later oscillated between 75% and 97.5% even at `1e-5`, and ended at 87.75%.
The corresponding fixed-`5e-5` run reached 98.25%. A one-way reaction to one
small validation drop can therefore lock in an unfavorable trajectory.

## Conclusion

The adaptive policy does not beat the simpler fixed `5e-5` policy. It reduces
some short-window volatility but introduces sensitivity to noisy trigger
measurements. The fixed `5e-5` setting remains the preferred late-curriculum
default.

Further tuning on this single formed checkpoint has diminishing scientific
value. The next important test is independent-initialization robustness: train
multiple models with the `hard400` scaffold and use `5e-5` at the 16-chunk
stage. This determines whether the complete formation-to-maintenance pipeline,
not merely one checkpoint, is reproducible.

