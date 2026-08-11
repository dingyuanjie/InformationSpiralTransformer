# Level 6.18.7 formal analysis

## Decision

**Level 6.18.7 completed successfully. Its preregistered classification is
`context_decodability_gain_not_behaviorally_causal`.** In precise statistical
language, this means that the context effect was **not confirmed as causal after
the registered three-test Holm correction**; it does not establish a zero
effect or prove that the context change is task-irrelevant.

The data show a consistent but boundary-level trend. At 16 chunks, transplanting
only the update-500 read context into the source raises held-out query accuracy
from 93.46% to 93.80% (+0.342 points; raw exact McNemar `p=0.0391`, Holm
`p=0.0781`). Switching from that condition to the full updated gate adds only
0.049 points (`p=1.0`). The complete route update reaches 93.85% (+0.391 points;
raw `p=0.0215`, Holm `p=0.0645`).

Thus the formal familywise threshold is narrowly missed, but the mechanism is
not gate cancellation. The changed read context reproduces 87.5% of the full
point-estimate gain; the gate adds little and does not reverse it. The reverse
restoration panel points in the same direction.

## Checkpoint and transplant integrity

Every validity check passed:

- exactly six route tensors and 24,896 parameters differ between checkpoints;
- no model or Probe parameter was updated during Level 6.18.7;
- returned persistent Memory was bitwise identical at 8, 12, and 16 chunks;
- context+gate, fused-feature, FFN-output, and query-hidden transplants exactly
  reproduced donor logits in both directions;
- maximum donor-logit discrepancy was `0.0`;
- donor prediction match rate was 100% for every required condition.

These exact positive controls show that final-query patching is correctly
placed and that the downstream FFN, residual/norm, and output head are identical
between source and update-500. The result is not a hook-order, partial-patch, or
checkpoint mismatch artifact.

## Baseline behavior

Each panel contains 2,048 identical held-out examples.

| Chunks | Source | Update 500 | Change | Paired bootstrap 95% CI | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 8 | 97.85% | 98.05% | +0.195 pp | [+0.049, +0.391] pp | 0.1250 |
| 12 | 96.39% | 96.44% | +0.049 pp | [-0.098, +0.195] pp | 1.0000 |
| 16 | 93.46% | 93.85% | +0.391 pp | [+0.098, +0.732] pp | 0.0215 |

The 16-chunk full update has nine source-wrong/update-correct examples and one
reverse discordance. Its uncorrected paired result is positive, but it belongs
to the preregistered three-contrast family and has Holm `p=0.0645`. The local
first-chunk control remains high and unchanged by final-query transplants:
97.80%, 97.95%, and 97.36% at 8, 12, and 16 chunks.

## Forward transplant: update into source

| Chunks | Source | Updated context through source gate | Updated gate activation only | Updated context+gate | Full update |
|---:|---:|---:|---:|---:|---:|
| 8 | 97.85% | 98.05% | 97.95% | 98.05% | 98.05% |
| 12 | 96.39% | 96.48% | 96.39% | 96.44% | 96.44% |
| 16 | 93.46% | 93.80% | 93.55% | 93.85% | 93.85% |

The context+gate, fused-feature, FFN-output, and query-hidden forward patches
all reproduce the full updated prediction vector exactly. The only clean
mechanism contrast before full reproduction is therefore the context-only
condition, followed by the conditional gate increment.

### Registered 16-chunk Holm family

| Contrast | Change | Discordances (left only / right only) | Raw p | Holm p | Confirmed at 0.05 |
|---|---:|---:|---:|---:|---:|
| updated context through source gate | +0.342 pp | 1 / 8 | 0.0391 | 0.0781 | no |
| updated gate after updated context | +0.049 pp | 0 / 1 | 1.0000 | 1.0000 | no |
| total source to update | +0.391 pp | 1 / 9 | 0.0215 | 0.0645 | no |

The updated gate does not cancel the updated context. Given the updated context,
switching to the updated gate corrects one additional example and harms none.
The small count cannot establish a positive gate contribution, but it rules out
the proposed pattern of a measurable negative gate effect in this panel.

The gate-activation-only row is a synthetic side control: the donor gate was
computed under the donor context and then combined with the receiver context.
Its +0.098-point result must not be interpreted as an independent gate-parameter
main effect.

## Reverse restoration: source into update

| Chunks | Update | Source context through updated gate | Source gate activation only | Source context+gate | Source |
|---:|---:|---:|---:|---:|---:|
| 8 | 98.05% | 97.90% | 98.00% | 97.85% | 97.85% |
| 12 | 96.44% | 96.39% | 96.48% | 96.39% | 96.39% |
| 16 | 93.85% | 93.55% | 93.75% | 93.46% | 93.46% |

At 16 chunks, restoring only the source context reduces update-500 accuracy by
0.293 points: seven examples are lost and one is gained (`p=0.0703`). Restoring
the source gate after the source context removes another 0.098 points (two lost,
none gained; `p=0.5`). Full source restoration loses 0.391 points with nine
losses and one gain (`p=0.0215`).

This reverse direction is corroborative rather than a second discovery family,
but it agrees with the forward decomposition: most of the small route effect is
carried by the context, while the gate supplies a smaller contribution in the
same direction.

## Cross-length mechanism pattern

The context point estimate grows with context length:

- 8 chunks: +0.195 points, four gains and no losses, raw `p=0.125`;
- 12 chunks: +0.098 points, two gains and no losses, raw `p=0.5`;
- 16 chunks: +0.342 points, eight gains and one loss, raw `p=0.0391`.

The effect is sparse in binary accuracy and strongest where the long-context
deficit is largest. This is compatible with a weak task-aligned improvement in
Memory reading, but the current familywise evidence is insufficient to call it
a confirmed causal rescue.

## Relation to Level 6.18.6

Level 6.18.6 found a +1.37-point increase in linear read-context decoding but
only +0.39 points in deployed behavior. Level 6.18.7 now shows:

1. the context change itself produces almost all of that +0.39-point behavioral
   movement when passed through the source gate;
2. the gate does not erase the movement;
3. the context's larger linear-decoding gain is only partly aligned with the
   deployed decision boundary;
4. binary accuracy is too sparse to confirm the small aligned component under
   the registered multiplicity correction.

Accordingly, `context_decodability_gain_not_behaviorally_causal` should be read
as **not confirmed behaviorally causal**, not as evidence that the context delta
contains no useful task direction.

## Scientific conclusion

Level 6.18.7 rejects the specific hypothesis proposed after Level 6.18.6 that
the fusion gate causally cancels a useful context gain. It instead supports a
weaker model:

- route training produces a large representational context shift;
- a small component is consistently aligned with correct long-context behavior;
- most of the extra linear decodability does not cross the deployed margin;
- the aligned component affects only about ten of 2,048 binary decisions, so it
  misses the conservative familywise threshold;
- broader route, FFN, norm, or head training is not yet justified.

There is no basis here for more identical optimizer steps, a gate-only rescue,
or opening seed 909.

## Next experiment

**Level 6.18.8 should diagnose the task-aligned read subspace with continuous
margin interventions, while keeping both checkpoints frozen.** At 16 chunks:

1. interpolate source-to-updated context with a preregistered dose curve;
2. measure correct-class logit margin and cross-entropy in addition to sparse
   argmax accuracy;
3. compute the source margin gradient with respect to read context and test
   whether the observed context delta has positive directional alignment;
4. compare the true delta with batch-rolled and norm-matched control deltas;
5. separate gradient-parallel and gradient-orthogonal context components and
   intervene on each.

This directly tests whether a small, reproducible task-aligned subspace exists
inside the much larger context shift. Only a positive controlled margin result
should authorize task-aligned read supervision or a new training boundary.
