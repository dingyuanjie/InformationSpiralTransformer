# Base Smoke 0.3.2 — Formal Analysis

## Gate result

**PASS.** Training continued from 0.3.1 step 1000 and stopped early after 400 additional steps (total step 1400) when two consecutive checks passed.

| condition | fixed 32 | held-out diagnostic |
|---|---:|---:|
| normal | 100.00% | 21.88% |
| zero-memory | 9.38% | 15.63% |
| reset-memory | 9.38% | 15.63% |
| roll-memory | 100.00% | 21.88% |
| zero-fast | 9.38% | 15.63% |

The fixed-set causal gap is 90.625 points. Zero-fast exactly equals reset and zero-all. Together with persistence-only subtraction and the exact no-history/reset invariants, this confirms that the frozen-Qwen adapter can learn to retrieve information through prior-chunk Fast Memory state.

## Optimization result

At total steps 1350 and 1400, fixed normal is 100% and the causal gap is 90.625 points, satisfying the consecutive-check rule. Separate conditioning works despite large raw gradients: Memory gradients range from about 1.16 to 165 and gate gradients from about 2.0 to 29.4, while independent clipping prevents one group from consuming the other's update budget. The FP32 residual scale changes smoothly to `-0.01001`.

Roll equals normal because slot attention is permutation invariant; zero-fast is the informative intervention.

## Boundary

Held-out normal is 21.88% (7/32), below four-way chance, and only 6.25 points above reset. No natural-language generalization is established. This pass proves architecture-level causal learnability on a fixed natural-language set, not a general long-context advantage.

## Decision

The architecture/gradient blocker is resolved for Fast Memory. Advance to a frozen-backbone generalization experiment with a large stream of unique training examples, validation-based checkpoint selection, multiple seeds, and paired normal/zero-fast evaluation. Keep Fast-only and persistence-only until held-out generalization appears; do not re-enable Slow/Episodic or partially unfreeze Qwen yet.
