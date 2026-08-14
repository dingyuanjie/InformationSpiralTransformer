# Level 6.19.2: frozen read-attention reachable-subspace audit

Level 6.19.1 showed that increasing attention to slots ranked by an independent
linear Memory Probe raises Probe margin but does not reliably improve deployed
decisions. An equal-L2 context-gradient control does improve decisions. This
level asks whether that useful context direction is reachable by the frozen
multi-head attention read at all.

## Frozen boundary

The source remains the formally passed Level 6.18.3 seed707 checkpoint at 16
chunks. The model, original Memory Probe, and Level 6.19 Memory/context probes
are frozen and fingerprinted before and after the run. The failed Level 6.18.9
candidate is excluded; protected tests and seed909 remain locked.

The final read is decomposed into eight heads, 32 attention probabilities per
head, projected slot values, and the shared output projection. The primary
population is again source errors for which the frozen Memory Probe is correct.
All gradient/oracle conditions use the true label and frozen deployed rival, so
they are mechanism tests rather than deployable inference methods.

## Conditions

- source: exact unmodified deployment path;
- Probe top-4 4x: FP32 analytic context delta corresponding to Level 6.19.1;
- gradient top-4 4x: four slots per head with highest deployed-margin
  attention-logit gradient;
- gradient KL oracle: exponential attention tilt maximizing first-order
  deployed margin under a matched per-head KL budget;
- negative-gradient and cross-example rolled-gradient KL controls;
- gradient L2 oracle: the same attention family matched to the Probe top-4
  context L2 dose;
- tangent context control: projection of the useful context gradient into the
  complete frozen attention value/output tangent span, matched in L2;
- unrestricted context control: full context gradient, matched in L2.

The KL reference is the per-example, per-head KL induced by Probe top-4 4x. If
a registered direction cannot attain that KL because it saturates at an
already-high-probability slot, every direction is capped to 95% of the common
reachable ceiling. The exact delivered budgets and cap fraction are recorded.

## Decision boundaries

The result distinguishes three cases:

1. a gradient-aligned finite attention redistribution recovers at least 50% of
   the unrestricted equal-L2 margin gain and passes all registered specificity
   controls: router-score obstruction;
2. the tangent span retains at least 80% of gradient energy and 80% of the
   unrestricted gain, but finite attention fails: simplex/budget limitation;
3. even the full frozen value/output tangent span is insufficient:
   value-composition/output-projection limitation.

## Numerical audit

Deployment uses the native bf16 `need_weights=False` attention path. The
interpretable decomposition uses explicit FP32 QK softmax and value/output
composition. The script verifies its internal FP32 reconstruction, the exact
deployed source downstream reconstruction, analytic gradients against
autograd, and all KL/L2 budgets. Each intervention adds only an analytic
context delta to the exact deployed source context, so the native-vs-explicit
kernel difference is not treated as an intervention effect.

### Numerical repair history

The first 4096-example formal attempt is preserved under
`formal/failed_numeric_v1/`. Direct fp32 KL accumulation produced impossible
negative KL ratios plus a rolled-control matching error just above tolerance.
Revision v2, preserved under `formal/failed_numeric_v2/`, started at native
autocast Q/K/V projections; its rare native-fused-SDPA versus explicit-softmax
context maximum was still 0.160. This established that the absolute maximum is
a cross-kernel discrepancy, not a projection error.

Revision v3 uses the scientifically relevant closed loop. The absolute
cross-kernel discrepancy is descriptive because no explicit absolute context
is substituted into deployment. Each intervention is
`explicit(updated) - explicit(source)` added to the exact deployed source; a
zero delta must reproduce the deployed result exactly, and the explicit
decomposition must reconstruct internally within `1e-5`. KL is accumulated in
FP64 with its theoretical non-negative bound, and the closest FP32-representable
bisection endpoint is selected. No model, data, dose, effect threshold, or
mechanism decision boundary changed.

The v3 full panel is preserved under `formal/failed_numeric_v3/`: every
closed-loop audit passed, but one rolled-control head (sample 2238 of 4096)
missed its KL target by `1.0401e-5`, just beyond the registered `1e-5` gate.
Revision v4 keeps the scalar KL bisection parameter in FP64 while evaluating
each candidate after quantization to the registered FP32 attention path. This
repairs the isolated solver-resolution failure without changing the target
budget or tolerance.

## Run

From `ist_v0_1`:

```powershell
python run_level6_19_2_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately **3-8 minutes**.
Progress is written to `experiments/level6_19_2/formal/progress.json`.

Formal outputs:

- `preregistration.json`;
- `result.json`, `summary.json`, and per-example `predictions.json`;
- `attention_reachable_subspace.png`;
- `ANALYSIS.md`, completed after the formal run.

After the archived numerical failure, rerun the repaired protocol once with:

```powershell
python run_level6_19_2_local.py --force
```

Do not otherwise use `--force` unless intentionally replacing a completed
formal result.
`--smoke-test` moves to a separate seed range and is never scientific evidence.
