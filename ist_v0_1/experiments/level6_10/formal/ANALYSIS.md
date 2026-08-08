# Level 6.10 analysis: frozen-state memory tomography

## Main result

Frozen IST states remain almost perfectly linearly decodable at every context
length. The apparent 16-chunk probe collapse was a stale-decoder problem, not
loss of linearly represented target information.

Mean test accuracy across three models:

| Chunks | Behavior | Original probe | Refit mean linear | Best layer | Best slot | Layer concat | All-state linear | MLP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 99.28% | 99.35% | 99.25% | 99.35% | 99.48% | 99.35% | 99.09% | 99.32% |
| 4 | 98.89% | 98.89% | 98.83% | 98.99% | 99.15% | 99.28% | 98.86% | 98.99% |
| 8 | 99.09% | 98.11% | 98.93% | 99.32% | 99.48% | 99.28% | 98.93% | 99.12% |
| 16 | 96.09% | 41.80% | **99.09%** | 99.15% | 99.38% | 99.38% | 98.99% | 99.28% |

At 16 chunks, simply refitting the same general class of mean-pooled linear
decoder raises accuracy from 41.8% to 99.1%. A nonlinear decoder is unnecessary
and does not outperform the best linear views.

## Layer localization

The best mean and concatenated layer is layer index 2 (the third/final layer)
for all 12 model x context-length combinations. At 16 chunks the layer-specific
result is especially sharp:

| Seed | Layer 0 mean | Layer 1 mean | Layer 2 mean |
| ---: | ---: | ---: | ---: |
| 606 | 5.57% | 6.54% | 98.34% |
| 808 | 6.84% | 6.54% | 99.51% |
| 1001 | 6.93% | 6.74% | 99.61% |

The first two layers are at the 6.25% chance level, while the final layer is
nearly perfect. Functional target memory is therefore localized by layer rather
than evenly distributed through the hierarchy.

## Slot distribution

Every best individual slot also comes from layer 2. More importantly, the code
is not concentrated in one special slot. At 16 chunks:

- seed 606 layer-2 slot mean is 97.57%, range 93.16--98.63%;
- seed 808 layer-2 slot mean is 99.31%, range 98.14--99.71%;
- seed 1001 layer-2 slot mean is 99.52%, range 99.02--99.80%.

Thus target information is redundantly replicated across most or all 32 slots
of the final layer. Mean pooling itself does not destroy the code: a newly fit
mean decoder reaches 99.1%.

## Why the original probe failed

The original probe was trained during earlier curriculum stages and then
frozen while memory updates and task training continued. It remains accurate at
2--8 chunks but becomes misaligned with the final 16-chunk representation for
seeds 808 and 1001. The target remains linearly present, but its coordinate
basis or scale drifts outside the frozen decoder's training distribution.

This explains the Level 6.8 and Level 6.9 discrepancy without invoking a hidden
nonlinear code. Probe accuracy measured with a stale decoder is not an invariant
property of information content.

## Decoder complexity

Full all-layer/all-slot concatenation and the MLP provide no meaningful gain.
The compact final-layer mean or even one final-layer slot is sufficient. The
slightly lower all-state linear result is consistent with unnecessary
high-dimensional features reducing sample efficiency.

## Architectural conclusions

1. Persistent target information at long context is carried predominantly in
   the final layer's memory.
2. It is highly redundant across the final layer's 32 slots.
3. The representation remains linearly decodable through 16 chunks.
4. Task-head accuracy (96.1%) is lower than post-hoc state decodability (about
   99.1--99.4%), so the memory contains more usable target information than the
   current behavioral readout extracts.
5. Future diagnostics should refit on held-out frozen states or track
   representation drift; a permanently frozen early probe is not reliable.

## Recommended next experiment

Tomography establishes decodability, not causal contribution of each layer or
slot. The next focused experiment should intervene selectively:

- zero one memory layer at a time;
- preserve only one layer at a time;
- mask increasing fractions of final-layer slots;
- compare fixed versus randomly selected slot subsets.

This will test whether the final-layer localization and slot redundancy are
causal properties of the model's read path, rather than properties visible only
to post-hoc probes.

