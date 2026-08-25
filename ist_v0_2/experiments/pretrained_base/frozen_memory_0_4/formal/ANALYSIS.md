# Frozen Memory 0.4 — Formal Analysis

## Result

The three-seed generalization gate **fails**.

| condition | accuracy | Wilson 95% |
|---|---:|---:|
| Base full context | 97.92% | 95.94--98.94% |
| IST normal | 25.78% | 21.66--30.38% |
| IST zero-fast | 26.30% | 22.15--30.92% |
| IST reset | 26.30% | 22.15--30.92% |

Normal is not above four-way chance and is slightly below zero-fast. Paired normal vs zero-fast: 1 improved, 3 harmed, 380 ties, difference -0.52 points, McNemar exact `p=0.625`. Normal is 72.14 points below Base (`p≈8.24e-84`).

Per-seed held-out normal is 22.66%, 31.25%, and 22.66%. Validation-selected causal gaps are 0, 0, and 3.125 points. There is no stable seed-level positive signal.

## Learning diagnosis

Unlike fixed-set continuation, every step uses unseen examples. The zero-initialized residual gate remains near zero: about `1e-5` for seeds 313/42 and `-9e-5` for seed 2026. Consequently the persistence delta has negligible behavioral amplitude and Memory receives weak/inconsistent learning signal. Training loss remains near four-way entropy (roughly 1.3--1.6), and validation predictions barely change across 1000 steps.

This is an optimization dead zone specific to unique-stream training: at exact zero scale, Memory gradients begin at zero; subsequent stochastic gate gradients change sign across unrelated samples before a stable retrieval representation forms. The fixed-set experiment escaped this dead zone through repeated aligned examples, but that mechanism does not transfer to a non-repeating stream.

## Decision

No held-out natural-language Memory generalization is established. Do not add more identical zero-gate steps, re-enable hierarchy, or unfreeze Qwen. Next test a preregistered nonzero but small persistence scale (magnitude learned in 0.3.2, approximately 0.01), held fixed during an initial Memory warmup, with distillation monitoring short-context preservation. This gives Fast Memory direct gradients without importing fixed-set Memory weights. After warmup, optionally unfreeze only the scalar gate.
