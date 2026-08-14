# Level 6.19.3: head-budget and signed-value tomography

Level 6.19.2 localized the frozen read-access gap: a finite gradient-aligned
softmax tilt recovered only 41.61% of the equal-L2 unrestricted context gain,
while the complete frozen value/output tangent span recovered 95.10%. This
level separates two explanations for that gap:

1. the useful finite dose is assigned to the wrong read heads;
2. the useful tangent direction requires zero-sum signed value mixtures that
   leave the non-negative attention simplex.

## Frozen boundary

The source is still the formally passed Level 6.18.3 seed707 checkpoint at 16
chunks. The original Memory Probe and both Level 6.19 linear probes are frozen
and fingerprinted. The failed Level 6.18.9 candidate is excluded, optimizer
search remains closed, protected tests are unopened, and seed909 stays locked.

The formal panel uses 4,096 newly generated examples at dataset seed 6193300,
disjoint from Level 6.19.2. The registered primary population is source errors
whose frozen persistent-Memory Probe remains correct.

## Equal-dose conditions

All main conditions use the per-example context L2 induced by the Level 6.19.1
Probe top-four 4x intervention:

- finite shared L2: the Level 6.19.2 gradient attention tilt;
- finite head-budget L2: an eight-dimensional non-negative finite-softmax
  Oracle, initialized independently from uniform, linearized NNLS, square-root
  NNLS, and a peaked best-head allocation, optimized on the exact finite
  context, and finally rematched to the registered L2 dose;
- signed affine L2: a minimum-coefficient, per-head zero-sum signed mixture in
  the complete frozen projected-value/output tangent span;
- negative, cross-example rolled, and head-permuted signed controls;
- unrestricted context-gradient positive control;
- eight one-head-only and eight leave-one-head-out signed projections.

All optimized directions use the true label and frozen source rival. They are
mechanism tests, not deployable inference methods.

## Registered decisions

- If finite head-budget recovery reaches 80% of unrestricted gain and
  significantly beats shared-dose finite attention, classify a head-budget
  allocation obstruction.
- Otherwise, if signed-affine recovery reaches 80% and beats source,
  shared-dose finite attention, negative, rolled, and head-permuted controls
  after Holm correction, classify a signed-affine simplex obstruction.
- Otherwise retain the head/subspace interaction as unresolved.

The head-only and leave-one-out panel localizes which heads carry unique useful
directions. It does not select a trainable checkpoint.

## Numerical protocol

The script inherits the passed Level 6.19.2 v4 closed-loop path. Finite
attention interventions add
`explicit(updated) - explicit(source)` to the exact native source context.
The explicit decomposition, analytic gradient, L2 doses, frozen fingerprints,
and closed-loop identity are all gated. Signed conditions are constructed
directly in the same projected-value/output basis and independently L2-gated.

## Run

From `ist_v0_1`:

```powershell
python run_level6_19_3_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately **6-15 minutes**.
Progress is written to `experiments/level6_19_3/formal/progress.json`.

Formal outputs:

- `preregistration.json`;
- `result.json` and compact `summary.json`;
- per-example `predictions.json`;
- `head_signed_tomography.png`;
- `ANALYSIS.md`, completed after the formal run.

Use `--smoke-test` only for implementation checks. Smoke results use a separate
seed range and are not scientific evidence. Do not use `--force` unless
intentionally replacing an incomplete formal result.
