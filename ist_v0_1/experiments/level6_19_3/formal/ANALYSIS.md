# Level 6.19.3 formal analysis

## Decision

- **Registered classification:** `signed_affine_simplex_obstruction` within the
  preregistered intervention family.
- The frozen source reached **92.3584%** accuracy on 4,096 fresh examples. It
  made 313 errors, of which **257/313 (82.1086%)** remained correct under the
  frozen persistent-Memory Probe.
- On those 257 registered primary examples, finite shared-dose attention
  recovered **39.57%** of the equal-L2 unrestricted deployed-margin gain.
- Reallocating the finite softmax dose independently across eight read heads
  raised recovery to **51.47%**, but failed the registered 80% closure gate.
- The equal-L2 signed-affine value mixture recovered **94.66%** and passed all
  five Holm-corrected specificity tests.
- The registered next boundary is therefore to isolate the smallest sufficient
  head set and test a gated residual/signed-value read. No checkpoint was
  selected or produced by this label-aware mechanism experiment.

## Integrity

All registered numerical and state-integrity gates passed.

| Audit | Result | Gate |
|---|---:|---:|
| Exact native source downstream reconstruction | 0 maximum absolute error | exact |
| Explicit FP32 internal reconstruction | 9.53674e-6 | <= 1e-5 |
| Updated-minus-source attention delta closure | 4.76837e-6 | <= 1e-5 |
| Analytic versus autograd attention gradient | 1.10944e-7 | <= 1e-6 |
| Reproduction of the shared finite family | 9.53674e-7 | <= 1e-5 |
| Shared finite-attention L2 match | 1.66893e-6 | <= 1e-3 |
| Head-budget finite-attention L2 match | 2.38419e-7 | <= 1e-3 |
| Signed-condition L2 match | 4.29153e-6 | <= 1e-5 |

Model states, probe states, and parameter fingerprints were unchanged. The
failed Level 6.18.9 candidate was excluded, the protected tests remained
unopened, seed909 remained locked, and optimizer search remained closed. The
formal run used dataset seed 6193300, analysis seed 6193400, and the frozen
Level 6.18.3 seed707 16-chunk checkpoint.

## Primary equal-L2 panel

The table below is restricted to the 257 Memory-decodable source errors. Every
condition was matched to the same per-example context-space L2 dose (mean
0.86420). Accuracy therefore means the fraction of these initially wrong
examples corrected by the fixed deployed decoder, not a deployable test-set
claim.

| Condition | Accuracy | Deployed margin gain | Context-Probe margin gain | Context L2 | Attention KL |
|---|---:|---:|---:|---:|---:|
| Finite shared L2 | 4.280% (11/257) | +0.06718 | +0.08565 | 0.86427 | 0.02414 |
| Finite head-budget L2 | 5.058% (13/257) | +0.08738 | +0.09804 | 0.86452 | 0.10970 |
| Signed affine L2 | 10.117% (26/257) | +0.16072 | +0.06754 | 0.86428 | n/a |
| Negative signed L2 | 0.000% (0/257) | -0.15840 | -0.06757 | 0.86432 | n/a |
| Rolled signed L2 | 0.778% (2/257) | -0.00294 | +0.00629 | 0.86447 | n/a |
| Head-permuted signed L2 | 0.000% (0/257) | -0.00086 | +0.01699 | 0.86450 | n/a |
| Unrestricted context L2 | 10.117% (26/257) | +0.16979 | +0.06245 | 0.86420 | n/a |

The head-budget Oracle significantly improved over the shared finite dose by
**+0.02020 deployed-margin units** (95% CI [0.01748, 0.02305], sign-flip
`p=9.999e-5`), but that improvement was far too small to close the registered
gap. It changed recovery from 39.57% to only 51.47%.

## Registered signed-affine specificity

All contrasts use the registered Memory-decodable-error population and the
fixed deployed margin. Holm correction covers the five signed-direction
contrasts as one family.

| Contrast | Mean advantage | 95% CI | Holm p | Positive pairs | Pass |
|---|---:|---:|---:|---:|---:|
| Signed vs source | +0.16072 | [0.14526, 0.17841] | 0.000500 | 100.00% | yes |
| Signed vs finite shared | +0.09354 | [0.08519, 0.10275] | 0.000500 | 98.44% | yes |
| Signed vs negative signed | +0.31912 | [0.28978, 0.35279] | 0.000500 | 100.00% | yes |
| Signed vs rolled signed | +0.16366 | [0.14703, 0.18342] | 0.000500 | 99.61% | yes |
| Signed vs head-permuted signed | +0.16159 | [0.14469, 0.18092] | 0.000500 | 99.61% | yes |

Direction, example identity, and head identity were all necessary for the
signed effect. The negative direction reversed the benefit, while rolling the
direction across examples or permuting it across heads reduced recovery to
approximately zero.

## Geometry and head tomography

The complete frozen read-attention tangent had mean rank **55.99** (minimum
50) and captured **89.78%** of the unrestricted context-gradient energy. The
eight-dimensional head-allocation family captured only **27.13%**. Across the
reported minimum-coefficient signed representation, 45.13% of coefficients
were negative, mean negative mass was 5.893, and every registered
example/head instance was flagged as unable to retain the full signed scale
inside the non-negative simplex.

| Head | Entropy | Rank | Grad. energy | Finite allocation | Finite 1st-order gain | Signed 1st-order gain | Head-only gain | Leave-one-out gain | Unique loss | Neg. mass | Neg. fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.609 | 7.996 | 0.1514 | 1.299 | 0.01232 | 0.01736 | 0.06467 | 0.15405 | 0.00667 | 6.380 | 0.440 |
| 1 | 2.307 | 7.996 | 0.1113 | 1.773 | 0.00548 | 0.01059 | 0.05689 | 0.15693 | 0.00379 | 5.199 | 0.477 |
| 2 | 3.016 | 7.992 | 0.1595 | 0.767 | 0.01206 | 0.02180 | 0.06958 | 0.15323 | 0.00750 | 5.398 | 0.422 |
| 3 | 2.705 | 8.000 | 0.1428 | 1.178 | 0.00886 | 0.01516 | 0.06412 | 0.15594 | 0.00479 | 5.357 | 0.451 |
| 4 | 2.804 | 7.992 | 0.1680 | 1.127 | 0.01108 | 0.02295 | 0.06784 | 0.15618 | 0.00454 | 6.156 | 0.449 |
| 5 | 2.822 | 7.992 | 0.1576 | 0.680 | 0.00733 | 0.02423 | 0.06844 | 0.15456 | 0.00616 | 6.017 | 0.483 |
| 6 | 2.972 | 7.934 | 0.1826 | 0.280 | 0.01463 | 0.02244 | 0.07126 | 0.15222 | 0.00851 | 6.325 | 0.453 |
| 7 | 2.832 | 8.000 | 0.1975 | 0.896 | 0.01513 | 0.02546 | 0.07373 | 0.15483 | 0.00589 | 6.310 | 0.436 |

The head-only and leave-one-out conditions were independently rematched to the
full registered L2 dose and are descriptive. A single head recovered only
35.4%-45.9% of the complete signed gain. Removing any one head retained
94.7%-97.6%, and the largest unique loss was only 0.00851 (head 6). Head 7 was
strongest alone, but no head was individually necessary. The useful signed
direction is therefore distributed and strongly redundant, not a single-head
switch.

## Behavioral transitions

| Population / condition | Corrections | Regressions | Resulting accuracy |
|---|---:|---:|---:|
| Memory-decodable errors: finite shared | 11 | 0 | 4.280% |
| Memory-decodable errors: finite head-budget | 13 | 0 | 5.058% |
| Memory-decodable errors: signed affine | 26 | 0 | 10.117% |
| Memory-decodable errors: unrestricted context | 26 | 0 | 10.117% |
| All 313 source errors: finite shared | 11 | 0 | 3.514% |
| All 313 source errors: finite head-budget | 13 | 0 | 4.153% |
| All 313 source errors: signed affine | 27 | 0 | 8.626% |
| All 313 source errors: unrestricted context | 27 | 0 | 8.626% |
| Full 4,096: finite shared | 11 | 0 | 92.627% |
| Full 4,096: finite head-budget | 13 | 0 | 92.676% |
| Full 4,096: signed affine | 27 | 0 | 93.018% |
| Full 4,096: negative signed | 0 | 15 | 91.992% |
| Full 4,096: rolled signed | 2 | 1 | 92.383% |
| Full 4,096: head-permuted signed | 0 | 2 | 92.310% |
| Full 4,096: unrestricted context | 27 | 0 | 93.018% |

All 313 confidence-matched source-correct examples remained correct under the
finite shared, finite head-budget, signed-affine, and unrestricted conditions.
The negative, rolled, and head-permuted controls caused 15, 1, and 2
regressions respectively. These accuracy transitions are mechanism upper
bounds because all optimized directions used the true label and frozen source
rival.

## Scientific conclusion

Level 6.19.3 rejects the narrow explanation that Level 6.19.2 failed merely
because one common finite dose was allocated uniformly across read heads.
Per-head finite optimization was real and statistically beneficial, but still
left almost half of the equal-L2 unrestricted margin gain unrecovered. In the
same frozen read-value tangent basis, allowing zero-sum signed mixtures nearly
closed the gap and survived direction, example, and head-identity controls.

The result supports a **distributed signed-value read-access obstruction** in
the registered family: the frozen Memory contains decodable information, and
the useful downstream direction exists across multiple read heads, but the
tested non-negative finite-softmax dose family cannot express enough of it.
This is stronger than a head-budget account and consistent with a need for an
explicit residual or signed-value read path.

There are two important limits. First, the intervention is label-aware and is
not a deployable predictor. Second, the registered finite head-budget Oracle
optimized eight head multipliers along specified gradient-aligned softmax
paths; it did not exhaust every arbitrary point in the full per-head attention
simplex. Likewise, negativity of the reported minimum-coefficient affine
solution does not by itself prove that every algebraically equivalent
coefficient representation is infeasible. Consequently the registered class
name must not be expanded into a theorem that all possible simplex-attention
reads fail.

## Next experiment

Follow the frozen diagnosis boundary with **Level 6.19.4: minimal-head signed
read and gated residual test**:

1. enumerate all non-empty head subsets at the same per-example L2 dose and
   identify the smallest subsets that retain at least 80% and 90% of the full
   signed-affine gain;
2. include an exact constrained-simplex feasibility/control audit so the
   conclusion remains scoped correctly;
3. train only a small label-free gated residual read on the development split,
   comparing non-negative gating, signed gating, matched-parameter residual,
   shuffled-memory, rolled-example, and head-permuted controls;
4. keep the seed707 trunk and all existing probes frozen, keep optimizer search
   closed, and do not open seed909 or protected tests until the preregistered
   development gates pass.

This next level should determine whether the distributed Oracle mechanism can
be compressed into a small head subset and converted into a genuine
input-conditioned read, rather than remaining a label-aware causal upper
bound.
