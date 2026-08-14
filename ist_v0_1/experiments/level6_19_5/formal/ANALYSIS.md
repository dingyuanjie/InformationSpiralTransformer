# Level 6.19.5 formal analysis

## Decision

- Integrity passes.
- Registered classification: `joint_calibration_coupling_bottleneck`.
- The frozen Level 6.19.4 router recovers 21.35% of the full signed-Oracle
  margin gain on the new primary population, again below 25%.
- Oracle dose + frozen learned direction recovers 29.72%; frozen learned dose
  + Oracle direction recovers 79.05%. Both cross the registered 25% boundary,
  while their original joint output does not.
- A directly supervised signed direction distiller recovers 30.71% at Oracle
  dose. Its matched-initialization residual control recovers 22.90%.
- Label-free observables strongly predict the error state and Oracle dose.
- Keep the trunk, seed909, and protected tests locked. Open at most the one
  factorized composition test specified in `../NEXT_STAGE.md`.

## Integrity

All registered integrity checks pass:

- frozen trunk, existing probes, and all three parent routers retain exact
  fingerprints;
- every frozen parameter remains frozen;
- parent-router fingerprints remain unchanged through the experiment;
- signed and residual direction probes each have 8,897 parameters;
- their initial trainable-parameter SHA-256 fingerprints match exactly;
- maximum Oracle L2 error is `6.68e-6`, below `1e-5`;
- train, validation, and formal seeds are disjoint;
- all four probes use their fixed final epoch;
- validation selects neither a checkpoint nor architecture;
- the formal panel is generated only after all probe weights are frozen;
- the failed Level 6.18.9 candidate is excluded;
- seed909 and protected tests remain unopened;
- model and optimizer search remain closed.

The offline BF16 query-tail replay differs from the native logits by at most
`0.0625` on each split, matching the registered parent replay precision. Every
condition is evaluated as a differential update around the native source
logits, so the source condition itself remains exact.

## Formal population

The one-shot formal panel contains 4,096 samples:

- source accuracy: 92.6758%;
- source errors: 300;
- Memory-decodable source errors: 263 (the primary population).

## Frozen dose-direction hybrids

Oracle components in this table use the target label and are causal diagnostic
ceilings, not deployable readers.

| Condition | Primary margin gain | Oracle recovery | Primary corrections | Full accuracy | Full corrections / regressions |
|---|---:|---:|---:|---:|---:|
| frozen learned dose + learned direction | +0.03589 | 21.35% | 7/263 | 92.8711% | 8 / 0 |
| Oracle dose + learned direction | +0.04997 | 29.72% | 11/263 | 92.9688% | 12 / 0 |
| learned dose + Oracle direction | +0.13292 | 79.05% | 19/263 | 93.1641% | 20 / 0 |
| Oracle dose + Oracle direction | +0.16814 | 100% | 23/263 | 93.2861% | 25 / 0 |
| probed dose + Oracle direction | +0.15868 | 94.38% | 22/263 | 93.2617% | 24 / 0 |
| Oracle dose + signed distilled direction | +0.05164 | 30.71% | 11/263 | 92.9443% | 11 / 0 |
| Oracle dose + residual distilled direction | +0.03851 | 22.90% | 8/263 | 92.8955% | 9 / 0 |

The frozen joint reader improves full accuracy by `+0.1953` percentage points
(bootstrap 95% CI `[+0.0732,+0.3418]` points; McNemar p=`0.0078125`) with eight
corrections and no regressions. This confirms a reproducible useful effect,
but it does not satisfy the registered Oracle-recovery threshold.

Both cross-hybrids pass 25%. Therefore neither frozen component is individually
incapable: the direction is just sufficient at Oracle dose, and the learned
dose is more than sufficient with Oracle direction. Their simultaneous output
falls below threshold, which triggers the pre-registered joint-coupling
classification. Quantitatively, direction quality is the tighter component,
but it is not an isolated hard failure.

## Held-out error-state observability

The 11,745-parameter classifier sees only the Level 6.19.4 router observables;
its formal inference receives no target label or rival class.

| Metric | Result |
|---|---:|
| primary prevalence | 6.42% |
| AUROC | 0.9593 |
| average precision | 0.7206 |
| precision in the top 263 scores | 64.64% |
| fixed-prevalence lift | 10.07x |

At a top-k size equal to the true primary count, 170 of 263 selected examples
are primary. The error opportunity is therefore strongly visible in the
label-free router inputs; the failure cannot be attributed to an absence of
error-state information.

## Held-out Oracle-dose observability

The fixed dose regressor also has 11,745 parameters and receives the same
label-free observables.

| Metric | Result |
|---|---:|
| target mean | 0.9457 |
| prediction mean | 0.9201 |
| MAE | 0.2392 |
| RMSE | 0.3680 |
| R2 | 0.6276 |
| Spearman correlation | 0.7561 |
| calibration intercept | 0.0435 |
| calibration slope | 0.9806 |

When combined with the label-aware Oracle direction, the predicted dose
recovers 94.38% of the Oracle deployed-margin gain and produces 24 corrections
with no regressions. Dose is therefore highly observable and close to
calibrated. This arm remains non-deployable because its direction is Oracle.

## Held-out direction observability and basis control

The signed and residual direction probes have matched initial trainable
weights, equal parameter counts, the same data and optimizer, and differ only
in their fixed output basis.

| Direction | Primary Oracle cosine | First-order ratio | Positive alignment | Deployed recovery at Oracle dose |
|---|---:|---:|---:|---:|
| signed memory-value basis | 0.2861 | 0.2865 | 83.27% | 30.71% |
| fixed residual basis | 0.2253 | 0.2263 | 85.55% | 22.90% |

Direct cosine supervision raises the signed basis above the 25% deployed
boundary, whereas the matched residual control remains below it. This is the
first fixed, matched-initialization direction diagnostic in this chain to show
a signed memory-value advantage. It is evidence within this registered basis
comparison, not a claim over every possible residual parameterization.

## Training audit

All probes run exactly 20 epochs and use the final epoch without selection.
Final validation losses are `1.0643` (classifier), `0.3184` (dose), `0.5050`
(signed direction), and `0.5291` (residual direction). The classifier's weighted
BCE validation loss is not used as a probability-calibration selector; its
formal rank metrics are reported independently.

## Scientific conclusion

Level 6.19.5 resolves the Level 6.19.4 ambiguity:

1. The label-free router inputs contain strong information about when an error
   opportunity exists and how large the Oracle intervention should be.
2. Dose estimation is not the limiting component; its label-free probe nearly
   reproduces the Oracle-dose ceiling when direction is supplied.
3. Signed direction is harder, but it is observable above the registered 25%
   boundary under direct direction supervision and it outperforms the matched
   residual basis.
4. The original joint router underperforms components that are separately
   sufficient. The registered conclusion is therefore a supervision and
   calibration coupling problem, not missing Memory information.

No fully label-free factorized candidate has yet been tested. The label-aware
hybrids and component ceilings do not constitute a deployable IST improvement.

## Next experiment

Proceed only to the frozen Level 6.19.6 single factorized-composition test in
`../NEXT_STAGE.md`. Do not tune its formula on this formal panel, open seed909,
or reopen model/optimizer search.
