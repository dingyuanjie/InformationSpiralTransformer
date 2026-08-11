# Level 6.18.6 formal analysis

## Decision

**Level 6.18.6 completed successfully and classifies the Level 6.18.5 route
update as `mixed_or_small_effect`.** There was no Python, CUDA, hook, checkpoint,
or incomplete-run error.

The update-500 route is not a recovery. At 16 chunks it changes the
`memory_context` representation substantially and raises its independently
refitted held-out linear decoding accuracy by 1.37 percentage points, but the
gain is not preserved by fusion: fused-feature decoding decreases by 0.59
points, query-hidden decoding increases by only 0.29 points, and deployed task
accuracy increases by only 0.39 points. The deployed change is not significant
by the exact paired McNemar test (`p=0.125`).

The scientific result is therefore more specific than the Level 6.18.5
validation failure: the selected routing tensors do learn a changed and weakly
more decodable Memory-read context, but the change is attenuated or cancelled
before it becomes useful query behavior.

## Checkpoint boundary audit

The audit passed exactly:

- six changed tensors;
- 24,896 changed parameters;
- every change under `blocks.2.memory_read.*` or
  `blocks.2.memory_fusion_gate.*`;
- no change to the original Memory Probe, persistent-Memory writer, FFN,
  normalization, output head, or any other model tensor.

| Tensor | Parameters | Maximum absolute change |
|---|---:|---:|
| `memory_read.in_proj_weight` | 12,288 | 0.01007 |
| `memory_read.in_proj_bias` | 192 | 0.00763 |
| `memory_read.out_proj.weight` | 4,096 | 0.00995 |
| `memory_read.out_proj.bias` | 64 | 0.00828 |
| `memory_fusion_gate.0.weight` | 8,192 | 0.00906 |
| `memory_fusion_gate.0.bias` | 64 | 0.00712 |

Persistent Memory and the pre-fusion feature were bitwise identical between
source and update-500 for every train, validation, and test split at 8, 12, and
16 chunks. The first numerically changed interface at every length was
`memory_context`, exactly as predicted by the parameter boundary.

## Held-out deployed behavior

Each row uses 1,024 identical test examples for both checkpoints.

| Chunks | Source query | Update-500 query | Change | Paired bootstrap 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 8 | 97.17% | 97.07% | -0.10 pp | [-0.29, 0.00] pp | 1.000 |
| 12 | 97.36% | 97.46% | +0.10 pp | [0.00, 0.29] pp | 1.000 |
| 16 | 91.80% | 92.19% | +0.39 pp | [0.10, 0.78] pp | 0.125 |

At 16 chunks there were four source-wrong/update-correct examples and no
reverse discordances. That sparse pattern explains why the percentile
bootstrap interval excludes zero while the exact two-sided McNemar test does
not reject the null. The conservative conclusion follows the exact paired
test: this is suggestive directional movement, not a confirmed behavioral
gain.

The local first-chunk control was unchanged at every length, including 97.75%
at 16 chunks. The original frozen Memory Probe was also exactly unchanged, as
expected from persistent-Memory invariance.

## Interface tomography

All entries below are independently refitted linear-Probe test accuracies on
1,024 held-out examples. Source and update fits at a given interface use a
matched optimizer seed.

| Interface | 8 source | 8 update | 12 source | 12 update | 16 source | 16 update |
|---|---:|---:|---:|---:|---:|---:|
| persistent Memory | 98.44% | 98.44% | 98.34% | 98.34% | 97.66% | 97.66% |
| pre-fusion feature | 94.73% | 94.73% | 93.26% | 93.26% | 89.94% | 89.94% |
| read context | 94.24% | 95.80% | 94.92% | 94.73% | 90.43% | 91.80% |
| fusion gate | 94.63% | 95.51% | 95.51% | 95.90% | 91.60% | 91.80% |
| fused feature | 95.70% | 95.80% | 94.34% | 93.95% | 91.70% | 91.11% |
| FFN output alone | 68.95% | 67.48% | 78.12% | 77.25% | 59.47% | 60.25% |
| query hidden | 96.97% | 96.29% | 96.88% | 96.97% | 91.70% | 91.99% |
| refit on deployed logits | 95.12% | 94.73% | 95.51% | 95.70% | 89.06% | 89.16% |
| deployed argmax | 97.17% | 97.07% | 97.36% | 97.46% | 91.80% | 92.19% |

At 16 chunks, the detailed source-to-update changes were:

| Interface | Change | Paired bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|
| persistent Memory | 0.00 pp | [0.00, 0.00] | 1.000 |
| pre-fusion feature | 0.00 pp | [0.00, 0.00] | 1.000 |
| read context | +1.37 pp | [+0.20, +2.54] | 0.0336 |
| fusion gate | +0.20 pp | [-0.29, +0.68] | 0.6875 |
| fused feature | -0.59 pp | [-1.17, -0.10] | 0.0703 |
| FFN output alone | +0.78 pp | [+0.10, +1.46] | 0.0386 |
| query hidden | +0.29 pp | [-0.10, +0.78] | 0.3750 |
| refit on deployed logits | +0.10 pp | [-0.20, +0.39] | 1.000 |

The raw read-context and FFN-output tests have `p<0.05`, but neither survives a
Holm correction across the complete 24-test interface-by-length family. Across
that family, only the 8-chunk read-context increase (+1.56 points) and the
8-chunk FFN-output decrease (-1.46 points) survive Holm correction. They point
in opposite directions and do not produce an 8-chunk behavioral improvement.
Thus the route update has real representational effects, but no consistent
cross-length useful effect.

## Same-example representation shift

At 16 chunks, the update is large at the selected route and progressively
attenuates downstream:

| Interface | Mean relative L2 change | Mean cosine similarity |
|---|---:|---:|
| read context | 19.07% | 0.9811 |
| fusion gate | 14.17% | 0.9893 |
| fused feature | 8.27% | 0.9963 |
| FFN output | 7.18% | 0.9984 |
| query hidden | 3.13% | 0.9992 |
| deployed logits | 2.94% | 0.9993 |

This is not a no-op optimizer. It is a strong route-level representation
change whose magnitude is compressed before the output. Persistent-Memory
decoding remains 97.66%. The source Memory-to-read-context gap is 7.23 points;
the update reduces it to 5.86 points. Yet the persistent-Memory-to-query-hidden
gap closes by only 0.29 points, from 5.96 to 5.66.

The automated JSON reports the largest scalar-decoding drop at
`fused_feature -> ffn_output`. That row must not be interpreted as proof that
the FFN destroys the task code. `norm2` receives the residual sum
`x + ffn(fused_feature)`, not the FFN output alone; query-hidden accuracy rises
back to about 92%. The mechanistically relevant finding is the failure to carry
the read-context improvement through fusion and the residual path, not the low
linear decodability of the isolated FFN branch.

## Causal Memory controls

The 16-chunk causal panel used 512 identical examples per condition.

| Condition | Source query | Update-500 query | Change | McNemar p |
|---|---:|---:|---:|---:|
| intact | 93.55% | 93.95% | +0.39 pp | 0.500 |
| reset | 5.86% | 5.86% | 0.00 pp | 1.000 |
| zero | 5.47% | 5.47% | 0.00 pp | 1.000 |
| batch roll | 6.64% | 6.45% | -0.20 pp | 1.000 |

The local control remained 98.44% in every condition. Both checkpoints retain
the same strong causal dependence on correct cross-chunk Memory. Route training
did not create a shortcut and did not reduce dependence on persistent Memory.

## Scientific conclusion

Level 6.18.6 establishes five points:

1. Level 6.18.5 really changed only the intended route; it was not a failed or
   frozen optimizer execution.
2. Persistent Memory remains high-information and exactly unchanged.
3. The update first acts at `memory_context` and sometimes improves its linear
   readability, including +1.37 points at 16 chunks.
4. That advantage is not reliably preserved by the fusion/residual pipeline;
   query-hidden and deployed gains remain small and statistically unconfirmed.
5. The result neither supports more identical routing updates nor justifies
   opening seed 909 or declaring a cross-initialization mechanism.

This strengthens the earlier Memory-to-query routing diagnosis while rejecting
the narrower claim that the final `memory_read + fusion_gate` parameter pair is
by itself sufficient for recovery.

## Next experiment

**Level 6.18.7 should be a frozen bidirectional activation-transplant test, not
a broader optimizer run.** On identical 16-chunk examples, transplant updated
activations into the source model and source activations into update-500 at:

1. read context only;
2. fusion gate only;
3. read context plus gate;
4. fused feature entering the FFN;
5. FFN output;
6. final query hidden.

Paired source, update, forward-transplant, and reverse-restoration predictions
will determine whether the weak context gain is causally cancelled at fusion,
inside the FFN/residual interaction, or only at the output. Only that result
should authorize a registered `fusion + FFN/norm2` training boundary.
