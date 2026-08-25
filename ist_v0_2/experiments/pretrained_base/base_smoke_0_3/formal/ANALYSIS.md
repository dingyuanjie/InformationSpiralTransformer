# Base Smoke 0.3 — Formal Analysis

## Result

The strict gate fails, but this is the first positive cross-chunk causal signal in the pretrained track:

| condition | fixed 32 | held-out diagnostic |
|---|---:|---:|
| normal | 53.13% | 18.75% |
| zero-memory | 9.38% | 18.75% |
| reset-memory | 9.38% | 18.75% |
| roll-memory | 53.13% | 18.75% |
| zero-fast | 43.75% | 18.75% |
| zero-slow | 53.13% | 18.75% |
| zero-episodic | 53.13% | 18.75% |

Normal exceeds zero/reset by 43.75 points. Because the persistence-only subtraction removes the query-local adapter path and reset exactly reproduces the no-history path, this difference is evidence that historical state is behaviorally useful on the fixed set. It does not meet the preregistered 95% accuracy / 50-point gap and does not generalize.

## Mechanism

- Both hard invariants pass with exact zero logit error.
- Router collapses to Fast from the first recorded check (`p_fast = 1.0`). Slow and Episodic probabilities are around `1e-18`.
- Fast gradients grow from 0.18 at step 50 to 3.18 at step 600. Slow and Episodic gradients remain around `1e-21` and `1e-20`.
- The residual scale remains small (`-0.00620`) but receives a large gradient, confirming an active persistence path.
- Roll has no effect, consistent with permutation-invariant slot reading.

`zero_fast` only reduces accuracy by 9.38 points rather than matching reset. Inspection shows that the intervention zeros Fast read memory but does not remove `fast_writer`'s state-dependent `base_feature`. Therefore this component ablation is incomplete and the present result cannot partition the Fast mechanism more finely.

## Decision

Persistence-only architecture has partial causal learnability, unlike 0.2, but the full gate fails and held-out remains below chance. Next: repair `zero_fast` so it cuts both Fast read and state-dependent base feature, lock a Fast-only diagnostic (rather than training dead Slow/Episodic routes), and continue the same fixed-set test with a better-conditioned residual gate. Do not claim natural-language generalization yet.
