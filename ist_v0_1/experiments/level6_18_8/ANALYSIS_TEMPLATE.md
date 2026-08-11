# Level 6.18.8 formal analysis

## Decision

- classification:
- integrity passed:
- true context-delta margin effect:
- true minus batch-roll effect:
- true minus random-direction effect:
- registered next boundary:

## Integrity audit

Confirm the six-tensor checkpoint boundary, exact persistent-Memory invariance,
exact alpha-1 context reconstruction, and finite nonzero per-example gradients.
Also confirm that the differentiable source-context hook reproduces source
logits exactly before interpreting its gradient.

## Dose curve

| Alpha | Accuracy | Fixed-rival margin | Decision margin | Cross-entropy |
|---:|---:|---:|---:|---:|
| 0.00 | | | | |
| 0.25 | | | | |
| 0.50 | | | | |
| 0.75 | | | | |
| 1.00 | | | | |

Report the paired per-example margin slope and whether the mean margin is
monotonic across the preregistered doses.

## Holm-corrected primary family

| Contrast | Estimate | 95% CI | Raw sign-flip p | Holm p | Pass |
|---|---:|---:|---:|---:|---:|
| true margin change | | | | | |
| true minus batch roll | | | | | |
| true minus random | | | | | |

## Gradient alignment

Report mean `gradient dot true_delta`, its positive fraction, cosine alignment,
control contrasts, and correlation with the observed alpha-1 margin change.

## Parallel and orthogonal interventions

The projection uses test labels and is mechanism-only. Compare the true delta,
gradient-parallel component, and gradient-orthogonal component without calling
the projection a deployable method.

## Scientific conclusion

Distinguish continuous task alignment from binary accuracy and distinguish true
delta specificity from a generic equal-norm perturbation effect.

## Next experiment

Follow only `analysis.diagnosis.registered_next_boundary` in `result.json`.
