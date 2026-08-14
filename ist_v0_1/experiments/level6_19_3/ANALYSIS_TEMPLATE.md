# Level 6.19.3 formal analysis

## Decision

- registered classification:
- source errors / Memory-decodable errors:
- finite shared-dose recovery:
- finite head-budget recovery:
- signed-affine recovery:
- registered next boundary:

## Integrity

Report exact native downstream reconstruction, internal explicit-FP32
reconstruction, closed updated-minus-source attention deltas, analytic versus
autograd gradient agreement, equal-L2 errors, frozen fingerprints, checkpoint
exclusion, protected-test lock, seed909 lock, and optimizer-search lock.

## Primary equal-L2 panel

| Condition | Deployed accuracy | Deployed margin gain | Context Probe margin gain | Context L2 | Attention KL |
|---|---:|---:|---:|---:|---:|
| finite shared L2 | | | | | |
| finite head-budget L2 | | | | | |
| signed affine L2 | | | | | n/a |
| negative signed L2 | | | | | n/a |
| rolled signed L2 | | | | | n/a |
| head-permuted signed L2 | | | | | n/a |
| unrestricted context L2 | | | | | n/a |

## Registered specificity

Report all five Holm-corrected signed-affine deployed-margin contrasts and the
finite head-budget versus shared-dose contrast.

## Head tomography

For heads 0-7 report entropy, rank, gradient energy, finite allocation, finite
and signed first-order contribution, one-head-only gain, leave-one-out gain,
unique loss, signed negative mass, and fraction of signed weights below zero.

## Behavioral transitions

Report corrections and regressions on Memory-decodable errors, all source
errors, the full panel, and confidence-matched source-correct cases. Treat all
label-aware accuracy as a mechanism upper bound.

## Scientific conclusion

Distinguish head-budget allocation from the need for signed affine value
mixtures. Do not call an oracle deployable, reopen optimizer search, or produce
a checkpoint.

## Next experiment

Follow only `analysis.diagnosis.registered_next_boundary` in `result.json`.
