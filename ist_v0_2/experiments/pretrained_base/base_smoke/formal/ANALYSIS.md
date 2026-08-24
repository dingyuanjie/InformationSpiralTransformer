# Pretrained 0.5B Base Smoke — Formal Analysis

## Integrity

- Complete with Qwen/Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Backbone: 494,032,768 frozen parameters.
- Trainable hierarchical Memory: 28,970,376 parameters.
- 50 Memory-only steps; 16 held-out examples per distance and condition.

## Result

| distance | Base full context | IST normal | IST zero/reset/roll | component zeros |
|---:|---:|---:|---:|---:|
| 512 | 100.00% | 81.25% | 81.25% | 81.25% |
| 1K | 100.00% | 25.00% | 25.00% | 25.00% |
| 2K | 100.00% | 25.00% | 25.00% | 25.00% |

Four-way chance is 25%. The pretrained Base solves every tested full-context example. The IST branch loses 18.75 points in one chunk and falls to chance as soon as retrieval crosses a chunk boundary.

All zero/reset/roll and Fast/Slow/Episodic-zero conditions exactly match IST normal at every distance. Therefore the trained Memory has no demonstrated behavioral causal effect.

## Mechanistic diagnosis

The adapter replaces the pretrained final hidden representation with a randomly initialized hierarchical-Memory feature before the frozen language head. It is not identity-preserving at initialization. The 512 regression is consequently evidence that the adapter damages pretrained representations, not evidence that persistent state is impossible. Training was also unstable (recorded losses 0.32--4.94), and 50 single-example steps are insufficient for a 28.97M-parameter adapter.

## Decision

Do not advance this adapter unchanged to long context or partial unfreezing. The next version must use a zero-initialized residual Memory delta so step-zero IST exactly reproduces Base logits, verify that equality numerically, and then train with batches plus a language-preservation/distillation objective. Only after short-context preservation passes should cross-chunk causal Memory be evaluated.
