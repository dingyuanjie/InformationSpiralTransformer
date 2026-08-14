# Level 6.19.4 formal analysis

## Decision

- registered classification:
- 80% minimal head set:
- 90% minimal head set:
- exact target-simplex feasibility:
- signed-router Oracle recovery:
- registered next boundary:

## Integrity

Report exact source reconstruction, equal-L2 Oracle errors, convex solver
convergence, frozen fingerprints, equal router parameter counts, disjoint split
seeds, diagnostic isolation, failed-candidate exclusion, protected-test lock,
seed909 lock, and optimizer-search lock.

## All-subset tomography

Report the best subset at each cardinality, every subset meeting the 80% and
90% thresholds, the registered selected sets, and their full-signed recovery.
Do not call any label-aware subset deployable.

## Exact simplex audit

Report feasible fraction, mean/median/maximum relative residual, two-start
delta gap, projected-gradient mapping, and both convergence gates. Scope the
result to representability of the exact registered signed target.

## Router training

For signed, non-negative, and matched residual routers report parameter count,
best epoch, validation task loss, distillation loss, accuracy, and mean gate.
Confirm that inference uses no target label or rival class.

## Fresh diagnostic panel

| Condition | Full accuracy | Primary margin gain | Primary correction rate | Context L2 | Gate |
|---|---:|---:|---:|---:|---:|
| source | | | | | |
| signed router | | | | | |
| non-negative router | | | | | |
| matched residual router | | | | | |
| shuffled-memory signed | | | | | |
| rolled-delta signed | | | | | |
| head-permuted signed | | | | | |
| selected-subset Oracle | | | | | |
| full signed Oracle | | | | | |

## Registered specificity

Report all six Holm-corrected signed-router deployed-margin contrasts, full
accuracy noninferiority, router-to-Oracle recovery, corrections, regressions,
and confidence-matched source-correct behavior.

## Scientific conclusion

Separate the label-aware minimal-head and simplex findings from the fixed
label-free-inference router result. Do not call the Oracle a trained model or
open seed909/protected tests unless the registered router gate passes.

## Next experiment

Follow only `analysis.diagnosis.registered_next_boundary` in `result.json`.
