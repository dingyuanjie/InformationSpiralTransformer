# Level 6.18.8 formal analysis

## Decision

**Level 6.18.8 passes all preregistered primary tests and classifies the result
as `task_aligned_context_subspace_confirmed`.** The update-500 final-block read
context contains a small but genuine task-aligned direction at 16 chunks. Its
continuous margin benefit is positive, dose-dependent, and specific relative
to equal-norm batch-roll and random-direction controls.

This is a mechanism confirmation, not a model-recovery pass. Full-dose query
accuracy is 93.16%, still below the earlier 95% recovery threshold. The result
is limited to seed707, the 16-chunk task, and the Level 6.18.5 update-500
checkpoint; it does not establish seed909 or cross-initialization transfer.

## Integrity audit

Every registered validity check passed:

- the inherited checkpoint audit found exactly six changed tensors and 24,896
  changed route parameters;
- returned persistent Memory was bitwise identical between source and update;
- `source_context + true_delta` reconstructed the update context exactly at
  alpha 1, with maximum absolute error `0.0`;
- the differentiable source-context hook reproduced source logits exactly,
  with maximum absolute error `0.0`;
- all 2,048 per-example context gradients were finite and nonzero;
- no model or Probe parameter was updated.

The positive result is therefore not explained by checkpoint leakage, Memory
mutation, context reconstruction error, or a gradient-hook discrepancy.

## Dose curve

The source context was interpolated toward update-500 at the final query token
while the source gate, FFN, normalization, and output head remained frozen.

| Alpha | Accuracy | Fixed-rival margin | Decision margin | Cross-entropy |
|---:|---:|---:|---:|---:|
| 0.00 | 92.68% | 5.55409 | 5.55409 | 0.25974 |
| 0.25 | 92.92% | 5.56741 | 5.56727 | 0.25704 |
| 0.50 | 92.92% | 5.57874 | 5.57807 | 0.25453 |
| 0.75 | 93.07% | 5.58896 | 5.58756 | 0.25236 |
| 1.00 | 93.16% | 5.59821 | 5.59591 | 0.25038 |

Mean fixed-rival margin is monotonically non-decreasing across all registered
doses. The per-example linear dose slope is +0.04392 with 95% CI
[+0.03285, +0.05631] and two-sided sign-flip `p=0.00010`.

The endpoint changes from alpha 0 to 1 are:

- fixed-rival margin: +0.04412, 95% CI [+0.03299, +0.05632];
- dynamic decision margin: +0.04182;
- cross-entropy: -0.00937, 95% CI [-0.01211, -0.00682];
- accuracy: +0.488 points, 95% bootstrap CI [+0.195, +0.781]
  points, exact McNemar `p=0.00195`.

Ten source errors become correct and no source-correct example becomes wrong.
The alpha-1 context intervention changes 11 predictions in total; the remaining
change is between two incorrect classes.

Even alpha 0.25 produces a significant continuous margin increase (+0.01332,
`p=0.00010`) and lower cross-entropy, while binary accuracy remains too sparse
for its exact McNemar test (`p=0.0625`). This directly explains why Level 6.18.7
was statistically borderline under a correct/wrong endpoint.

## Holm-corrected primary family

The primary fixed-rival margin uses the source model's strongest incorrect
class as a fixed comparison for each example.

| Contrast | Estimate | 95% CI | Raw sign-flip p | Holm p | Pass |
|---|---:|---:|---:|---:|---:|
| true margin change | +0.04412 | [+0.03269, +0.05641] | 0.00010 | 0.00030 | yes |
| true minus batch roll | +0.05268 | [+0.04188, +0.06483] | 0.00010 | 0.00030 | yes |
| true minus random | +0.04545 | [+0.03352, +0.05813] | 0.00010 | 0.00030 | yes |

All three estimates are positive and all three pass the preregistered Holm
family. This meets the exact confirmation rule.

The average equal-norm batch-roll direction changes margin by -0.00856, 95% CI
[-0.01325, -0.00363]. The average equal-norm random direction changes margin by
-0.00132, with CI [-0.00294, +0.00020]. The true delta therefore does not work
merely because a context perturbation has the same norm. It is sample-specific
and direction-specific.

## Gradient alignment

At the unchanged source context, the frozen correct-versus-source-rival margin
gradient gives:

- mean `gradient dot true_delta`: +0.05013;
- 95% CI: [+0.03756, +0.06315];
- two-sided sign-flip `p=0.00010`;
- true-minus-batch-roll derivative: +0.05552;
- true-minus-random derivative: +0.05086;
- mean gradient/delta cosine: +0.05316;
- correlation between the first-order derivative and observed alpha-1 margin
  change: **0.98224**.

Only 49.17% of examples have a positive gradient/delta cosine, while the mean
directional effect is strongly positive. The population gain is therefore not
a uniform shift affecting most examples; positive-alignment examples have
larger magnitude than negative-alignment examples. This matches the sparse
argmax changes and should not be described as a universal per-example benefit.

## Parallel and orthogonal interventions

The per-example true delta was decomposed relative to the labeled source-margin
gradient. This analysis uses evaluation labels and is mechanism-only.

| Intervention | Accuracy change | Fixed-margin change | 95% CI | Margin p |
|---|---:|---:|---:|---:|
| full true delta | +0.488 pp | +0.04412 | [+0.03299, +0.05632] | 0.00010 |
| gradient-parallel component | +0.488 pp | +0.04669 | [+0.03572, +0.05815] | 0.00010 |
| gradient-orthogonal component | 0.000 pp | -0.00094 | [-0.00216, +0.00037] | 0.1534 |

The gradient-parallel intervention produces exactly the same prediction vector
as the full true-delta context intervention and the full update-500 model. It
recovers all ten additional correct predictions. The orthogonal intervention
produces exactly the source prediction vector and no accuracy change.

The orthogonal component slightly lowers average cross-entropy by 0.00090 but
does not improve the fixed-rival margin, decision margin, or any prediction.
This is a small calibration/non-target-class effect, not evidence that the
orthogonal subspace carries the confirmed decision benefit.

Because the projection uses the true label and per-example test gradient, it is
not a deployable routing rule. Its role is to localize the already confirmed
true-delta effect.

## Context versus the full route update

The full update-500 model reaches the same 93.16% accuracy and exactly the same
prediction vector as the alpha-1 context-only intervention. Its fixed-rival
margin gain is +0.05058 versus +0.04412 for context alone, a further +0.00646.

Thus the updated gate supplies a small continuous-margin increment but does not
alter any argmax decision in this panel. This independently confirms the Level
6.18.7 conclusion: the gate does not cancel the context benefit, and the
behaviorally relevant part of the route update is carried by the read context.

## Scientific conclusion

The combined Levels 6.18.6-6.18.8 now establish a coherent mechanism:

1. update-500 changes only the intended final Memory-read route;
2. persistent Memory remains unchanged and highly decodable;
3. most of the route's representational movement is not task-aligned;
4. a smaller direction inside the context delta reliably improves the deployed
   correct-class margin;
5. that direction beats equal-norm sample-mismatched and random controls;
6. its first-order gradient alignment predicts the observed effect with
   correlation 0.982;
7. the gradient-parallel component carries the prediction changes, while the
   orthogonal component does not;
8. the confirmed mechanism improves but does not recover 16-chunk behavior to
   95%.

Accordingly, the earlier Level 6.18.5 failure was not a complete absence of
learning. It learned a real task-aligned read direction, but the effect was too
small and too mixed with orthogonal movement to pass the stable recovery gate.

## Next experiment

**Level 6.18.9 should be a preregistered task-aligned read-supervision rescue,
with all modules except the final `memory_read` frozen.** It should:

1. start again from the formally passed Level 6.18.3 source, not update-500;
2. freeze `fusion_gate`, FFN/norm, output head, all Memory writers, embeddings,
   and lower blocks;
3. train only the four final `memory_read` tensors;
4. use balanced 8/12/16 batches;
5. optimize deployed correct-class margin together with a contrast requiring
   intact Memory to beat norm-matched batch-rolled Memory;
6. regularize context displacement from the source to suppress large
   task-orthogonal drift;
7. gate progress on disjoint continuous-margin validation before opening the
   existing accuracy tests;
8. retain exact persistent-Memory invariance and reset/zero/batch-roll causal
   checks.

This is the first broader learning step justified by the tomography. It should
remain a single registered training protocol, not reopen learning-rate or
optimizer search, and it must not open seed909 before seed707 passes its full
protected gate.
