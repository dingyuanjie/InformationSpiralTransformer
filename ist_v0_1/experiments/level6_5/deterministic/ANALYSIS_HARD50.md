# Level 6.5 deterministic hard50 confirmation

## Result

`hard50` used probe-loss weight 0.5 for the first 50 optimization steps and
zero probe loss thereafter. Its integrated supervision cost was 25 weighted
probe steps. Five deterministic initializations were tested.

| Seed | Final 2-chunk query | Maximum query | Local | Passed |
| ---: | ---: | ---: | ---: | ---: |
| 313 | 50.00% | 50.00% | 90.00% | no |
| 42 | 16.25% | 16.25% | 100.00% | no |
| 2026 | 76.25% | 76.25% | 96.25% | no |
| 7 | 2.50% | 15.00% | 98.75% | no |
| 1234 | 67.50% | 67.50% | 93.75% | no |

Success rate: **0/5**. All runs stopped at the 2-chunk formation gate.

## Interpretation

Fifty scaffold steps are insufficient for reliable formation under the fixed
3,000-step budget. The high local accuracies show that this is not a general
training failure. Three seeds ended far above the 6.25% chance level, so the
scaffold altered the optimization trajectory and often produced partial or
delayed formation, but it did not produce a stable 95% solution.

The earlier nondeterministic seed-313 pilot that reached 100% should be treated
as evidence that a 50-step trajectory *can* form, not as evidence that it forms
reliably. Deterministic multi-seed confirmation is the registered result.

## Decision

Do not search below 50 steps yet. The next search should expand upward to find
a reproducible formation region. Test longer hard-stop schedules and matched
linear annealing schedules on one deterministic seed, then confirm the first
promising region across five seeds.

