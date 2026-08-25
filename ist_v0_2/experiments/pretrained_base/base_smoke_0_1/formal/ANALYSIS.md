# Base Smoke 0.1 — Formal Analysis

## Integrity

- Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Frozen backbone: 494,032,768 parameters; trainable adapter: 28,970,377.
- Identity invariant passed exactly: initial maximum logit delta = 0.
- 200 steps and 32 held-out examples per distance/condition.

## Result

| distance | Base | IST normal | zero/reset/roll range |
|---:|---:|---:|---:|
| 512 | 100.00% | 93.75% | 93.75--100.00% |
| 1K | 96.88% | 21.88% | 21.88--25.00% |
| 2K | 100.00% | 25.00% | 25.00% |

Identity preservation fixes most of the prior one-chunk damage (81.25% -> 93.75%), but does not create cross-chunk retrieval. Four-way chance is 25%; IST is at/below chance at 1K and 2K while full-context Base remains 96.88--100%.

Normal never beats zero/reset/roll or a component-zero condition. At 512, zeroing Episodic restores Base accuracy from 93.75% to 100%, which is a small negative causal effect of the current Episodic branch. There is no positive persistent-Memory causal effect.

The learned residual scale remains tiny (`-0.00705`). Training is unstable and underdetermined: a 28.97M-parameter adapter received only 200 single-example steps, with final task losses still fluctuating around 1.2--2.2.

## Decision

The preservation problem is substantially improved, but the Memory-learning gate fails. Before adding data or unfreezing Qwen, run a controlled 1K overfit/gradient diagnostic on a fixed small set. If the adapter cannot reach near-100% training retrieval and show a normal-vs-zero gap, the architecture or gradient path is defective; if it can, the next problem is training scale/generalization.
