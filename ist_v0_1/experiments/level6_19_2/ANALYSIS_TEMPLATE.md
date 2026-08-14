# Level 6.19.2 formal analysis

## Decision

- registered classification:
- source errors / Memory-decodable errors:
- gradient KL oracle deployed-margin gain:
- equal-L2 attention recovery fraction:
- tangent gradient energy and recovery fraction:
- registered next boundary:

## Integrity and numerical audit

Confirm exact native downstream reconstruction, FP32 attention-decomposition
reconstruction within tolerance, analytic-versus-autograd gradient agreement,
matched KL/L2 budgets, frozen state fingerprints, and all checkpoint/test
locks. Report the native-bf16 versus explicit-FP32 context discrepancy
separately from intervention effects.

## Primary panel

| Condition | Deployed accuracy | Deployed margin | Context Probe accuracy | Context Probe margin | Context L2 | Attention KL |
|---|---:|---:|---:|---:|---:|---:|
| source | | | | | | |
| Probe top-4 4x | | | | | | |
| gradient top-4 4x | | | | | | |
| gradient KL oracle | | | | | | |
| negative-gradient KL | | | | | | |
| rolled-gradient KL | | | | | | |
| gradient L2 oracle | | | | | | |
| tangent context control | | | | | | n/a |
| unrestricted context control | | | | | | n/a |

## Registered gradient-KL specificity

Report the four Holm-corrected deployed-margin contrasts: gradient KL versus
source, Probe top-4, negative-gradient KL, and rolled-gradient KL.

## Reachable-subspace geometry

Report per-example tangent rank, projected gradient-energy fraction, attention
gradient norm, common versus Probe-reference KL, and the margin-gain recovery
fractions for gradient KL, equal-L2 gradient attention, and tangent projection
relative to unrestricted context.

## Behavioral transitions and controls

Report corrections/regressions on Memory-decodable errors, all source errors,
the full panel, and confidence-matched correct cases. Treat label-aware oracle
accuracy only as a mechanism upper bound.

## Scientific conclusion

Distinguish router scoring, finite simplex/KL constraints, and the linear span
of frozen per-head values plus output projection. Do not call any oracle a
deployable rescue or reopen optimizer search.

## Next experiment

Follow only `analysis.diagnosis.registered_next_boundary` in `result.json`.
