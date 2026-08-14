# Level 6.19.2 formal analysis

## Decision

**Level 6.19.2 passes the repaired closed-loop numerical protocol and localizes
the remaining read-access obstruction to finite attention geometry, not to a
missing direction in the frozen value/output tangent span.** The registered
classification is `finite_attention_simplex_or_budget_limitation`.

On 4,096 new 16-chunk examples, the frozen Level 6.18.3 seed707 source is
correct on 3,771/4,096 = **92.07%** and makes 325 errors. The frozen Level 6.19
Memory Probe is correct on 255/325 = **78.46%** of those errors; these 255
Memory-decodable source errors form the registered primary population.

At equal context L2, the best finite gradient-aligned attention tilt increases
the deployed correct-versus-source-rival margin by **+0.0748** and recovers
only **41.61%** of the unrestricted context-gradient gain, below the registered
50% sufficiency boundary. In contrast, projection of that context gradient
onto the complete frozen attention value/output tangent span gives **+0.1710**,
recovering **95.10%** of the unrestricted gain. The tangent span contains
**89.43%** of context-gradient energy on average. Thus the useful local
direction is substantially present in the value/output span, but the registered
finite attention redistribution cannot realize enough of it at equal dose.

The registered next boundary is:

> The value/output tangent span contains the useful direction, but finite
> attention redistribution at the registered budgets does not recover it;
> audit head-wise budgets and signed value mixtures.

All gradient and tangent conditions use the true label and frozen source rival.
They are mechanistic upper bounds, not deployable inference rules. No checkpoint
is produced and optimizer search remains closed.

## Integrity and numerical audit

The final result uses numerical revision
`delta_closed_loop_fp64_solver_v4`. Every registered integrity gate passes:

- the unmodified native downstream path reconstructs source logits exactly,
  maximum error `0.0`;
- the explicit FP32 value/output composition reconstructs the explicit FP32
  attention context within `7.63e-6`, below `1e-5`;
- the actual intervention identity
  `explicit(updated) - explicit(source)` closes within `5.98e-6`, below
  `1e-5`, before being added to the exact deployed source context;
- analytic attention-logit gradients agree with autograd within `1.15e-7`,
  below `1e-6`;
- worst equal-KL errors are `1.78e-7` for the positive, negative, and rolled
  directions, below `1e-5`;
- worst equal-L2 error is `1.67e-6`, below `1e-3`;
- all model and Probe parameters remain frozen and every state fingerprint is
  unchanged;
- the failed Level 6.18.9 candidate is excluded, protected tests are unopened,
  and seed909 remains locked.

The native fused-SDPA versus explicit-softmax absolute context maximum is
reported descriptively as `0.1600` (`0.1875` for a separate native-kernel
replay). This cross-kernel absolute difference is not an intervention effect:
the explicit absolute context is never substituted into deployment. Only the
closed explicit updated-minus-source delta is added to the exact native source,
and that path passes its direct gate above.

Three earlier full-panel attempts are retained under `failed_numeric_v1/`,
`failed_numeric_v2/`, and `failed_numeric_v3/` rather than overwritten. They
identified, respectively, negative FP32 KL roundoff, the irrelevance of an
absolute cross-kernel maximum, and one FP32 scalar-bisection resolution failure
at sample 2238. Revision v4 retains the KL tilt parameter in FP64, evaluates
each candidate on the registered FP32 attention path, and preserves the
original `1e-5` tolerance. No scientific effect threshold, model, data seed,
condition, or intervention dose was changed.

The confidence-matched control contains 325 source-correct examples. Mean
source confidence is 1.4731 on errors and 1.5844 on matched correct examples;
mean absolute matching distance is 0.1117 and the maximum is 0.3125.

## Primary Memory-decodable error panel

| Condition | Deployed accuracy | Deployed margin | Context Probe accuracy | Context Probe margin | Context L2 | Attention KL |
|---|---:|---:|---:|---:|---:|---:|
| source | 0.00% | -2.0022 | 49.02% | 0.6739 | 0.0000 | 0.00000 |
| Probe top-4 4x | 0.00% | -2.0145 | 49.41% | 0.7276 | 0.8727 | 0.12374 |
| gradient top-4 4x | 2.35% | -1.8834 | 49.41% | 0.8758 | 1.3548 | 0.18935 |
| gradient KL oracle | 3.14% | -1.8371 | 51.37% | 0.9149 | 1.7142 | 0.12374 |
| negative-gradient KL | 0.00% | -2.1507 | 40.39% | 0.4273 | 1.6805 | 0.12374 |
| rolled-gradient KL | 0.00% | -2.0069 | 44.31% | 0.6546 | 1.4919 | 0.12374 |
| gradient L2 oracle | 1.57% | -1.9274 | 50.20% | 0.7929 | 0.8726 | 0.02594 |
| tangent context control | 4.31% | -1.8312 | 49.80% | 0.7215 | 0.8727 | n/a |
| unrestricted context control | 4.71% | -1.8224 | 49.80% | 0.7163 | 0.8727 | n/a |

The Probe top-four intervention again moves the independent context Probe in
the expected direction (+0.0537 margin) but makes deployed margin worse by
-0.0124 and corrects no primary errors. Gradient alignment, rather than more
of the same Probe-ranked routing, is necessary.

The label-aware gradient top-four condition corrects 6/255 errors. The
matched-KL gradient oracle corrects 8/255; its reverse direction corrects none
and sharply harms both deployed and Probe margins, while another example's
rolled gradient is near null. This establishes direction-specific causal
control of the frozen read, but it does not establish a deployable router.

## Registered gradient-KL specificity

All four registered deployed-margin contrasts pass Holm correction:

| Contrast on 255 primary cases | Estimate | 95% CI | Raw p | Holm p | Positive fraction |
|---|---:|---:|---:|---:|---:|
| gradient KL vs source | +0.1651 | [+0.1238, +0.2209] | 0.00010 | 0.00040 | 100.0% |
| gradient KL vs Probe top-4 | +0.1774 | [+0.1312, +0.2418] | 0.00010 | 0.00040 | 100.0% |
| gradient KL vs negative KL | +0.3136 | [+0.2458, +0.4059] | 0.00010 | 0.00040 | 100.0% |
| gradient KL vs rolled KL | +0.1698 | [+0.1217, +0.2318] | 0.00010 | 0.00040 | 97.65% |

The reverse-direction condition changes primary deployed margin by -0.1485,
whereas the rolled direction changes it by -0.0047 with `p=0.577`. Equal KL
dose alone therefore does not explain the oracle gain. The gain is tied to the
correct per-example deployed-gradient direction.

## Reachable-subspace geometry

The registered recovery comparison is:

| Mechanistic condition | Primary margin gain | Fraction of unrestricted gain | Registered interpretation |
|---|---:|---:|---|
| gradient KL oracle | +0.1651 | 91.79% | unequal L2; specificity/dose control |
| gradient L2 oracle | +0.0748 | **41.61%** | finite attention fails 50% boundary |
| tangent projection, equal L2 | +0.1710 | **95.10%** | tangent direction is operational |
| unrestricted context, equal L2 | +0.1798 | 100.00% | positive control |

The gradient-KL oracle is strong but changes context by mean L2 1.7142, almost
twice the common equal-L2 dose 0.8727; its 91.79% number is not evidence of
equal-dose attention sufficiency. The registered attention-sufficiency decision
therefore correctly uses the gradient-L2 oracle, whose 41.61% recovery fails.

Across primary cases, the frozen value/output tangent rank averages 55.59 of
64 and has minimum 50. Its projection retains 89.43% of context-gradient
energy on average (minimum 68.74%). Despite that per-example tail, its mean
behavioral recovery is 95.10%, above the registered 80% boundary. Mean
attention-logit-gradient norm is 0.02897 and mean context-gradient norm is
0.20044.

The primary mean Probe reference KL and common matched KL are both 0.12374.
Only 1/255 primary cases is capped, to 75% of its Probe reference; the remaining
254 use the full reference. Across all 4,096 cases, 430 are capped, with minimum
fraction 0.50. Thus the primary result is not driven by broad dose clipping.

The tangent control is a signed local linear-span test. It may redistribute
head contributions with signs and amplitudes that no finite non-negative
attention probability vector can realize at the registered dose. Consequently,
the gap between 95.10% tangent recovery and 41.61% finite-attention recovery is
the evidence for the simplex/budget classification; it is not evidence that an
ordinary softmax router can directly implement the tangent oracle.

## Behavioral transitions and controls

| Population / condition | Accuracy | Corrections | Regressions | McNemar p |
|---|---:|---:|---:|---:|
| 255 primary errors, gradient top-4 | 2.35% | 6 | 0 | 0.03125 |
| 255 primary errors, gradient KL | 3.14% | 8 | 0 | 0.00781 |
| 255 primary errors, gradient L2 | 1.57% | 4 | 0 | 0.12500 |
| 255 primary errors, tangent | 4.31% | 11 | 0 | 0.00098 |
| 255 primary errors, unrestricted | 4.71% | 12 | 0 | 0.00049 |
| all 325 source errors, gradient KL | 3.08% | 10 | 0 | 0.00195 |
| all 325 source errors, tangent | 3.69% | 12 | 0 | 0.00049 |
| all 325 source errors, unrestricted | 4.00% | 13 | 0 | 0.00024 |
| all 4,096, gradient KL | 92.31% | 10 | 0 | 0.00195 |
| all 4,096, tangent | 92.36% | 12 | 0 | 0.00049 |
| all 4,096, unrestricted | 92.38% | 13 | 0 | 0.00024 |

On the 325 confidence-matched correct cases, gradient top-four, gradient KL,
gradient L2, tangent, and unrestricted conditions cause zero regressions.
Probe top-four and rolled-gradient KL each regress 2/325, while the deliberately
wrong negative-gradient direction regresses 13/325 (`p=0.00024`). The latter is
a directionality control, not a safety candidate.

Across the complete panel, Probe top-four corrects no error and regresses three
examples, reducing accuracy from 92.07% to 91.99%. Gradient KL raises accuracy
to 92.31% through 10 label-aware corrections with no regressions; tangent and
unrestricted controls reach 92.36% and 92.38%. These accuracy changes are
mechanistic upper bounds only: they use the true label to choose a favorable
direction and cannot be reported as a deployable model improvement.

## Scientific conclusion

Level 6.19.2 resolves the ambiguity left by Level 6.19.1:

1. A per-example deployed-gradient attention direction is causal and specific:
   it beats source, Probe targeting, its negative, and a rolled direction at
   equal KL.
2. Simply ranking four slots with the independent linear Memory Probe remains
   insufficient and can move the deployed margin in the wrong direction.
3. At equal L2, a finite softmax attention tilt realizes only 41.61% of the
   useful unrestricted context gain.
4. The complete frozen value/output tangent span realizes 95.10%, while
   retaining 89.43% of the gradient energy.
5. Therefore the useful direction is not missing from the frozen projected
   values plus output projection. It is largely present locally, but finite
   non-negative per-head attention mixing at the registered budgets cannot
   express enough of it.

This rules out `value_output_reachable_subspace_limitation` as the current
primary diagnosis and also rejects the stronger claim that router scores alone
are already sufficient. The supported boundary is narrower: **head-wise
simplex constraints, unequal head budgets, and/or the need for signed value
mixtures obstruct the finite read.**

The result does not authorize a trained checkpoint, reopening seed909, or a
new optimizer sweep. It is a frozen causal localization result.

## Next experiment

Proceed to **Level 6.19.3: head-wise budget and signed value-mixture tomography**.
Keep the Level 6.18.3 model, Level 6.19 probes, seed707 source, protected tests,
and seed909 frozen. On the same primary definition:

1. decompose the tangent oracle and finite gradient-L2 oracle into eight
   head-specific context contributions;
2. measure per head its source attention entropy, attainable positive/negative
   coefficient range, KL/L2 cost, gradient energy, and marginal deployed-margin
   gain;
3. use leave-one-head-out and one-head-only interventions to identify whether
   the 41.61% gap is concentrated in particular heads;
4. compare the ordinary simplex with registered signed-affine value mixtures
   at equal total context L2, then with head-wise L2 reallocation while holding
   total dose fixed;
5. include negative, rolled-head, and equal-dose controls and repeat all
   closed-loop numerical audits;
6. branch only after the frozen audit: if signed mixtures close the tangent
   gap, investigate a minimal gated residual/value-mixture read; if head-budget
   reallocation alone closes it, investigate a donor-free head-wise router
   budget rule.

The next level should remain diagnostic and label-aware. A deployable training
intervention is justified only after this head-wise mechanism is identified.
