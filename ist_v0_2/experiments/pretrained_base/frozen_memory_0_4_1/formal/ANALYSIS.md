# Frozen Memory 0.4.1 — Formal Analysis

## Result

The generalization gate fails.

| condition | accuracy | Wilson 95% |
|---|---:|---:|
| Base full context | 96.35% | 93.97--97.82% |
| IST normal | 28.13% | 23.86--32.82% |
| IST zero-fast/reset | 26.04% | 21.90--30.65% |

Normal exceeds zero-fast by 2.08 points, but paired evidence is not significant: 25 improved, 17 harmed, 342 ties, McNemar exact `p=0.280`. Normal's interval contains four-way chance.

Per-seed held-out effects are inconsistent: seed 313 has no gap, seed 42 is +9.38 points, and seed 2026 is -3.13 points. Validation-selected gaps are only 4.69%, 3.13%, and 0%. All selected checkpoints occur early in fixed-gate warmup (steps 100--200); later training does not improve validation.

## Diagnosis

The nonzero scale successfully removes the zero-gradient dead zone: Memory gradients are nonzero from step 1 and remain measurable. Nevertheless task loss stays around 1.3--1.8 and validation is mostly static. Optimization access alone is therefore insufficient.

The current distillation target is `adapter.last_base_logits`, produced by Qwen on the final 512-token Query Chunk only. That teacher cannot see the fact in the first Chunk. It preserves local language behavior but provides no dense target for reconstructing the full-context retrieval computation. The only cross-chunk supervision is four-way answer CE, which is sparse and noisy for learning a general write/read algorithm from non-repeating examples.

The writer also compresses every token in the fact/distractor Chunk without a direct salience or representation target. Fixed-set repetition can memorize this mapping, but unique-stream training does not discover a transferable selection rule.

## Decision

Do not increase identical steps or gate magnitude. Next use the frozen full-context Qwen as a teacher: run Qwen once on the complete 1K example and distill its final hidden state and/or full-vocabulary logits into the chunked IST student's final state. Retain answer CE, Fast-only persistence, zero-fast causality, and short-context preservation. This directly teaches the Memory path to approximate a computation the same backbone already performs successfully.
