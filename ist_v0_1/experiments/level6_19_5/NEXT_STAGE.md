# Proposed Level 6.19.6: one frozen factorized signed composition

## Boundary

Freeze the Level 6.18.3 seed707 trunk, all existing probes and parent routers,
and the final-epoch Level 6.19.5 classifier, dose probe, signed direction
distiller, and residual direction control. Perform no training, calibration,
architecture selection, threshold search, or checkpoint selection. Keep the
failed Level 6.18.9 candidate excluded, seed909 locked, protected tests closed,
and optimizer/model search closed.

Use one new 4,096-example formal diagnostic with seed `6196100`. Do not replay
or select on the Level 6.19.5 formal panel.

## Single repair candidate

The only repair candidate is the fully label-free factorized signed read:

```text
delta_factorized(x) = predicted_oracle_dose(x)
                      * signed_distilled_unit_direction(x)
```

Use the frozen Level 6.19.5 dose transform and its fixed `[0, 8]` cap exactly as
saved. Use the frozen signed direction distiller exactly as saved. The candidate
receives only the Level 6.19.4 router observables and never receives a target
label, rival class, Oracle dose, Oracle direction, or correctness flag.

The error-state classifier is reported as an enrichment audit but is not used
as a gate. This avoids choosing a probability calibration or threshold after
seeing Level 6.19.5. No second repaired candidate is allowed.

## Registered controls

- source;
- frozen Level 6.19.4 signed router;
- the one factorized signed candidate;
- factorized residual-basis control using the same predicted dose;
- factorized signed with shuffled memory observables;
- rolled factorized signed delta;
- head-permuted signed coefficients;
- full label-aware signed Oracle as the ceiling only.

Controls are not selectable candidates.

## Success gate

The factorized signed candidate passes only if all conditions hold:

1. at least 25% of the full signed-Oracle deployed-margin gain is recovered on
   fresh Memory-decodable source errors;
2. paired deployed-margin contrasts versus source, the frozen router, residual
   control, shuffled memory, rolled delta, and head-permuted coefficients are
   all positive after Holm correction at 0.05;
3. the lower 95% CI of full-panel accuracy change is no worse than -0.25
   percentage points;
4. all frozen-state, split, label-free-inference, and exact-L2 integrity gates
   pass.

If it passes, repeat the entire frozen factorized reader across independent
probe initializations before opening seed909. If it fails, stop this router
repair branch; do not introduce a second composition formula.
