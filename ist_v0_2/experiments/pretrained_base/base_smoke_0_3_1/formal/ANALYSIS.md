# Base Smoke 0.3.1 — Formal Analysis

## Result

The full preregistered gate fails because fixed normal is below 95%, but Fast persistence is now cleanly causal:

| condition | fixed 32 | held-out diagnostic |
|---|---:|---:|
| normal | 75.00% | 28.13% |
| zero-memory | 9.38% | 6.25% |
| reset-memory | 9.38% | 6.25% |
| roll-memory | 75.00% | 28.13% |
| zero-fast | 9.38% | 6.25% |

Normal exceeds reset/zero by 65.625 points, above the 50-point causal threshold. The repaired zero-fast exactly equals reset, so both Fast read and state-dependent Fast base feature are now removed. With persistence-only subtraction and exact identity/reset invariants, the improvement is attributable to prior-chunk Fast state rather than a query-local adapter path.

## Learning dynamics

Fixed normal rises from 21.88% at step 50 to 75.00% at step 1000, with the strongest sustained rise after step 600. Reset stays fixed at 9.38%. The causal gap passes 50 points from step 650 onward, but normal never reaches 95%.

The residual gate remains small (`-0.00946`) while its raw gradient is highly unstable (about 3.8--474). Fast gradient ranges from about 0.3 to 24.6. Because both are currently included in one global gradient clip, large scalar-gate gradients can consume the clipping budget and suppress Fast parameter learning. This is a concrete optimization defect.

Roll equals normal, which is expected for permutation-invariant slot attention and is not evidence against stored information.

Held-out normal is 28.13% versus 6.25% for reset, but with only 32 examples this is not a stable generalization claim and normal is not statistically established above four-way chance.

## Decision

Fast-only persistence is causally confirmed on the fixed set, but the 95% learnability gate remains open. Next use separate optimizer groups and clipping: lower FP32 gate learning rate with an independent gradient clamp, and clip Fast parameters separately. Continue from the 0.3.1 checkpoint rather than restarting. Do not re-enable Slow/Episodic or unfreeze Qwen yet.
