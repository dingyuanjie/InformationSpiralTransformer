# Proposed Level 6.19.5: router observability-supervision diagnosis

## Boundary

Keep the Level 6.18.3 seed707 trunk, all existing probes, and the three Level
6.19.4 routers frozen. Do not reopen architecture or optimizer search. Keep
seed909 and protected tests locked. Pre-register new, disjoint diagnostic
train/validation/formal seeds before generating data.

## Question

The Level 6.19.4 signed router recovers 23.31% of the full Oracle and fails only
the matched-residual specificity contrast in addition to the 25% recovery
gate. Determine which component is limiting label-free compilation:

1. intervention dose/gating is not observable from router inputs;
2. signed direction is not observable from router inputs;
3. the information is observable but the joint task-plus-distillation
   supervision fails to extract it;
4. the projected memory-value basis is not uniquely useful relative to a
   generic residual basis.

## Fixed diagnostic decomposition

On a new formal split, decompose each frozen router delta into unit direction
and scalar dose. Replay four signed conditions through the same frozen tail:

- learned dose + learned direction (the frozen Level 6.19.4 reader);
- Oracle dose + learned direction (direction ceiling);
- learned dose + Oracle direction (dose/gating ceiling);
- Oracle dose + Oracle direction (the registered full signed Oracle).

The hybrid arms are label-aware diagnostics and must never be described as
deployable readers.

Fit, with one fixed initialization and no model selection, three small
out-of-sample diagnostic probes using exactly the Level 6.19.4 router
observables:

- primary/error-state classifier: AUROC, AUPRC, and fixed-prevalence lift;
- Oracle-dose regressor: held-out R2, Spearman correlation, and calibration;
- Oracle-direction distiller: held-out cosine, first-order margin alignment,
  and deployed-margin recovery at Oracle dose.

Use an equal-parameter residual-direction distiller as the registered basis
control. Select no architecture from these results.

## Interpretation rule

- Oracle dose + learned direction passes 25%, but learned dose + Oracle
  direction does not: dose/gating is the dominant bottleneck.
- Learned dose + Oracle direction passes, but Oracle dose + learned direction
  does not: signed-direction observability/supervision is dominant.
- Both hybrids pass while the frozen combination fails: joint calibration or
  coupling is dominant.
- Neither hybrid passes while the full Oracle does: both components are
  limiting or the observables are insufficient; report probe ceilings to
  distinguish these cases.
- A residual distiller matching or exceeding the signed distiller again means
  the value-basis-specific deployment claim remains unsupported.

This level is diagnostic only. It cannot open seed909 or protected tests.
