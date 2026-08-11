# Level 6.18.8: task-aligned read subspace

Level 6.18.7 found a consistent but sparse 16-chunk context-transplant trend:
the updated read context accounted for 0.342 of the full 0.391-point accuracy
gain, but the registered Holm-corrected binary tests narrowly missed 0.05.
Level 6.18.8 replaces the low-power correct/wrong endpoint with continuous logit
margin while keeping both checkpoints completely frozen.

## Target and dose curve

The target is the final block's `memory_context` at the final query token. The
source context is interpolated toward the update-500 context at fixed doses:

`alpha = 0.00, 0.25, 0.50, 0.75, 1.00`.

The primary margin is the correct-class logit minus the strongest incorrect
source-model logit. Fixing the source rival makes the dose curve differentiable
and comparable across interventions. Dynamic decision margin, cross-entropy,
and argmax accuracy are secondary metrics.

## Registered null controls

For every example, the true context delta is compared with:

- eight within-batch rolled directions, each rescaled to the true delta norm;
- eight random directions, each rescaled to the true delta norm.

Three paired sign-flip tests form one Holm-corrected primary family:

1. true context delta raises fixed-rival margin;
2. its effect exceeds the average batch-roll effect;
3. its effect exceeds the average random-direction effect.

All three estimates must be positive with Holm `p < 0.05` to confirm a
task-aligned context subspace.

## Gradient diagnostics

The script computes the frozen source margin gradient with respect to each
example's read context and reports `gradient dot true_delta`. It also intervenes
with the gradient-parallel and gradient-orthogonal components of the true
delta.

These component diagnostics use evaluation labels. They localize mechanism but
are not deployable, are not model training, and are not part of primary
confirmation.

## Integrity

The run requires:

- exact persistent-Memory invariance between source and update 500;
- exact reconstruction of the updated query context at `alpha=1`;
- exact source-logit reproduction through the differentiable context hook;
- finite, nonzero per-example context gradients;
- the inherited six-tensor/24,896-parameter checkpoint boundary;
- no model or Probe updates.

## Run

From `ist_v0_1`:

```powershell
python run_level6_18_8_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 15-35 minutes.
Artifacts are written to `experiments/level6_18_8/formal/`:

- `preregistration.json`;
- `result.json`, `summary.json`, and `predictions.json`;
- `task_aligned_subspace.png`;
- the completed `ANALYSIS.md` based on `ANALYSIS_TEMPLATE.md`.

Only a controlled positive-margin result authorizes a later task-aligned read
supervision experiment. This level does not open seed 909 or update any model.
