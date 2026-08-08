# Level 6.5.1 analysis: 16-chunk optimization stability

## Controlled setup

All four variants loaded the same `hard400_seed313/stage3.pt` checkpoint,
restored its Adam moments, used the same deterministic data stream, froze the
probe, and used zero probe loss. Only the learning rate differed. Each variant
received 1,000 nominal training steps followed by 500 continued maintenance
steps at the same learning rate.

## Results

| LR | Peak query | First consecutive gate | Step 1000 | Step 1500 final | Maintenance tail min | Registered pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 93.75% | never | 78.75% | 77.00% | 73.75% | no |
| 5e-5 | 100.00% | 400 | 97.50% | 95.75% | 91.25% | yes |
| 2.5e-5 | 98.75% | 1100 | 93.75% | 97.25% | 95.00% | no* |
| 1e-5 | 98.75% | 450 | 95.00% | 96.75% | 92.50% | yes |

`2.5e-5` failed the preregistered rule only because it had not reached 95% at
the step-1000 boundary. It crossed the consecutive gate at step 1100 and then
finished with strong, stable performance. It is best described as delayed
convergence rather than long-context instability.

## Interpretation

The experiment isolates update magnitude as the cause of the Level 6.5
16-chunk degradation. The original 1e-4 setting did not merely converge more
slowly: it remained unstable and ended near 77% after 1,500 steps. Reducing the
learning rate by 2--10x preserved or recovered the transferred long-context
solution.

`5e-5` is the best primary setting under the current budget. It crossed the
gate earliest among the robust settings, reached 100%, and passed both the
step-1000 and post-maintenance criteria. `1e-5` is also robust but more
conservative. `2.5e-5` shows that fixed step boundaries can confound learning
rate comparisons with convergence time.

Probe accuracy remains near chance because the briefly trained probe was frozen
long before this stage. Behavioral query accuracy is the primary memory metric,
as defined in the corrected Level 6.5 protocol.

## Conclusion

The 16-chunk failure was an optimization-stability problem, not a memory
capacity limit. With a suitable late-curriculum learning rate, the model can
carry the scaffolded solution to 2,048 total tokens and retain approximately
96--97% query accuracy after 1,500 probe-free updates.

## Recommended Level 6.5.2

Confirm `5e-5` across multiple deterministic data-stream seeds from the same
stage-3 checkpoint, with `1e-4` as the destructive-update control. This tests
whether the stabilization is stream robust before returning to the harder
question of independently formed model initializations.

