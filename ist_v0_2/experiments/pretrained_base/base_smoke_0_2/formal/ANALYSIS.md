# Base Smoke 0.2 — Formal Analysis

## Result

The adapter reaches 100% on the 32 fixed training examples, but fails the causal overfit gate:

| condition | fixed train | held-out diagnostic |
|---|---:|---:|
| normal | 100.00% | 15.63% |
| zero-memory | 96.88% | 18.75% |
| reset-memory | 96.88% | 18.75% |
| roll-memory | 96.88% | 18.75% |
| zero-fast | 100.00% | 15.63% |
| zero-slow | 100.00% | 15.63% |
| zero-episodic | 100.00% | 18.75% |

The normal-minus-zero/reset causal gap is only 3.125 points, far below the required 50 points. Held-out is below four-way chance (25%). This is memorization without persistent-state dependence.

## Mechanism

- Router is effectively 100% Episodic (`p_episodic = 1.0`); Fast, Slow, and Forget probabilities are around `1e-10`.
- Slow write is around `6e-12` and Slow gradient norms are around `1e-20`; Slow is functionally dead.
- Router gradients are around `1e-9`, so the saturated routing decision cannot recover.
- Fast-writer gradients remain substantial (roughly 0.14--4.50) even though `p_fast` is near zero. This occurs because the existing hierarchical module always exposes `fast_writer`'s local `base_feature` to the adapter output independently of persistent routing.
- Episodic gradients are nonzero, but zero-Episodic does not reduce fixed-set accuracy, so they do not establish historical-state use.

## Conclusion

The architecture/gradient diagnostic fails. The adapter learned a query-local transformation that memorizes the fixed set, not cross-chunk Memory. Adding more examples or unfreezing Qwen would obscure this defect.

The next adapter must expose only a persistence delta: `Memory(hidden, historical_state) - Memory(hidden, reset_state)`. The local path cancels exactly, reset-state output reproduces Base, and any normal-vs-reset gain must arise from prior-chunk state. Router saturation should also be prevented or bypassed during this diagnostic.
