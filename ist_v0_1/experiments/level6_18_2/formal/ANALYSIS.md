# Level 6.18.2 formal analysis

## Decision

The preregistered diagnosis is **output-head alignment bottleneck**. At 12
chunks, the frozen IST state still contains high-quality target information in
both persistent Memory and the final query-token hidden state, but the trained
task output head under-reads that information.

This is a post-hoc mechanism result. It does not retroactively pass Level
6.18.1.

## Frozen behavior

The selected Level 6.18.1 diagnostic checkpoint is the failed 8-to-12 bridge at
step 1500. On fixed held-out evaluation:

| Metric | Accuracy |
|---|---:|
| 8-chunk task-head control (800 samples) | 95.75% |
| 12-chunk task head (1,024 samples) | 90.33% |
| 12-chunk original Memory probe | 93.55% |
| 12-chunk local control | 99.41% |

The length transition therefore costs the original task head 5.42 percentage
points, while local retrieval remains intact.

## Frozen tomography

| Refitted linear feature | Held-out accuracy |
|---|---:|
| Third-layer Memory mean | 96.48% |
| Third-layer all-slot concatenation | 98.05% |
| All-layer Memory means | 98.14% |
| All-layer/all-slot Memory | **98.63%** |
| Final query-token hidden | **96.97%** |
| Third-layer Memory + query hidden | 98.34% |
| All Memory + query hidden | 98.63% |

The best Memory decoder exceeds the trained task head by 8.30 points. The
refitted query-hidden decoder exceeds it by 6.64 points and trails the best
Memory decoder by only 1.66 points. Therefore:

1. persistent information is not lost at 12 chunks;
2. Memory-to-token routing is sufficiently functional to place linearly
   decodable target information in the final query token;
3. most of the observed behavioral deficit is downstream, at the trained
   output-head alignment;
4. adding query hidden to all Memory does not improve on all Memory alone, so
   there is no evidence of important complementary target information outside
   the Memory state;
5. third-layer Memory is already nearly sufficient: all layers improve over
   third-layer all-slot concatenation by only 0.59 points.

The old probe's 93.55% versus the refitted Memory decoder's 98.63% is another
representation-drift result, not evidence of missing information.

## Per-sample decoupling

The original task head makes 99 errors on the 1,024-example test set.

- all-Memory linear decoding corrects 88 of those 99 errors (88.89%);
- query-hidden linear decoding corrects 71 of 99 errors (71.72%);
- the task-head/all-Memory oracle union reaches 98.93%;
- only 11 examples are wrong for both the task head and all-Memory decoder.

This is direct sample-level evidence that the decodable state is substantially
better than the behavior exposed by the original output head.

## Causal control

| Condition | Query | Local |
|---|---:|---:|
| Intact | 90.33% | 99.41% |
| Reset Memory | 6.54% | 99.41% |
| Zero Memory | 6.45% | 99.41% |
| Batch-roll Memory | 5.66% | 99.41% |

The strongest disrupted condition is 6.54%, close to 16-class chance (6.25%),
and the intact-to-disrupted drop is 83.79 points. The causal gate passes. The
high-quality frozen representations therefore depend on sample-specific
cross-chunk Memory rather than local leakage or a dataset artifact.

## Next falsification test

Level 6.18.3 should perform a surgical output-head rescue on this same frozen
checkpoint:

1. freeze every IST parameter except `model.output`;
2. fit the output head using 12-chunk query targets only;
3. compare against an untouched checkpoint on disjoint 8-, 12-, and 16-chunk
   test sets;
4. require 12-chunk query at least 95% without reducing the 8-chunk control
   below 95%;
5. repeat reset, zero, and batch-roll controls after rescue.

If head-only fitting recovers 12-chunk behavior while retaining the 8-chunk
circuit and Memory causality, the output-head diagnosis is causally confirmed.
If it does not, the linear query-hidden result is decodable but not sufficient
for a stable shared readout, and the next target becomes final-block routing or
length-conditioned representation drift.

