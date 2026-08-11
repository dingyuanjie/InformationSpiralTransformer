# Level 6.18.5 formal analysis

## Decision

**Level 6.18.5 fails the preregistered stable routing gate.** This is not a
Python, CUDA, checkpoint, or backward-pass error. The script intentionally did
not open the protected test and causal datasets after the validation failure.

The result is a valid negative mechanism intervention: updating only the final
block's `memory_read` and `memory_fusion_gate` with 16-chunk query loss is not
sufficient, under this frozen protocol, to produce stable >=95% behavior at 8,
12, and 16 chunks.

It does not refute the Level 6.18.4 routing diagnosis in general; it rejects
this particular parameter boundary and optimization protocol as a complete
rescue.

## Training execution audit

The optimizer completed all 500 registered updates. All six allowed routing
tensors changed:

| Tensor | Maximum absolute change |
|---|---:|
| `memory_read.in_proj_weight` | 0.01007 |
| `memory_read.in_proj_bias` | 0.00763 |
| `memory_read.out_proj.weight` | 0.00995 |
| `memory_read.out_proj.bias` | 0.00828 |
| `memory_fusion_gate.0.weight` | 0.00906 |
| `memory_fusion_gate.0.bias` | 0.00712 |

The optimizer state records step 500. The failure is therefore not a frozen
parameter, missing-gradient, resume, or no-op bug.

## Validation trajectory

For updates 1–400, the fixed 80-example screens were exactly:

- 8 chunks: 100.00%;
- 12 chunks: 98.75%;
- 16 chunks: 91.25%.

At updates 425–500, the 16-chunk screen increased only to 92.50%. This activated
the larger confirmation panel, whose repeated fixed results were:

| Length | Confirmation accuracy | Required |
|---:|---:|---:|
| 8 chunks | 96.09% | >=95% -- pass |
| 12 chunks | 94.53% | >=95% -- fail by 0.47 pp |
| 16 chunks | 91.80% | >=95% -- fail by 3.20 pp |

No checkpoint achieved one complete confirmation, so the stability streak
remained 0/2. Consequently `routing_best.pt` and `routing_stable.pt` were not
created; `routing_latest.pt` correctly contains the update-500 diagnostic
state.

## Optimization behavior

The accumulated training minibatches were highly variable:

- observed training query accuracy ranged from 62.5% to 100%;
- recorded loss ranged from approximately 0.002 to 1.23;
- the last four evaluations reported 100% minibatch accuracy;
- fixed 16-chunk validation remained at 92.5%.

This is a training-to-validation disconnect. The restricted routing modules can
change and fit sampled query batches, but those changes do not generalize into
the missing stable Memory-to-query transfer. The almost constant validation
curve also argues against merely adding a small number of identical updates.

## What was not evaluated

Per the fail-closed preregistration, the following protected artifacts do not
exist and must not be inferred from the screen:

- 2,048-example paired 8/12/16 tests;
- formal parameter mutation audit on a selected stable checkpoint;
- formal per-chunk Memory invariance audit;
- protected 16-chunk intact/reset/zero/batch-roll causal test;
- `routing_rescued_checkpoint.pt` and final result figure.

The architectural Memory-invariance property was verified in the pre-run smoke
test, but there is no formally accepted routing checkpoint to which the final
gate can be applied.

## Scientific conclusion

Level 6.18.4 showed a significant 4.59-point gap between all-Memory decoding
and query-hidden decoding. Level 6.18.5 shows that directly optimizing the
last-layer MHA read and fusion gate does not automatically close that gap.

Several mechanisms remain compatible with the combined evidence:

1. the relevant loss occurs after `memory_read`, inside fusion, the frozen FFN,
   normalization, or residual interaction;
2. the query-token route needs coordinated changes beyond MHA/gate parameters;
3. the query-only, batch-2 accumulated objective is underconstrained and learns
   sample-specific corrections rather than a general read map;
4. the 4.59-point Memory/query gap is distributed across several final-block
   interfaces rather than concentrated in the two selected modules.

The result does not justify an unregistered learning-rate increase, more steps,
or opening seed 909.

## Next falsification test

Level 6.18.6 should return to a frozen diagnostic: final-block routing
tomography at 8, 12, and 16 chunks for both the Level 6.18.3 source and the
update-500 Level 6.18.5 latest checkpoint.

Capture and refit held-out linear decoders on:

1. final-layer persistent Memory;
2. query-position `memory_read` context;
3. fusion-gate output;
4. the fused `memory_feature` entering the FFN;
5. FFN output;
6. final `norm2`/query-token hidden;
7. deployed output logits.

Use identical examples for source and update-500 models and measure where
decodability first drops, and whether route training moved that interface at
all. Only after identifying the destructive interface should a broader module
such as `memory_read + fusion_gate + FFN/norm2` be considered for intervention.

