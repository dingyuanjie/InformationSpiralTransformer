# Level 6.19.4: minimal heads and a gated signed read

Level 6.19.3 showed that reallocating a finite non-negative attention dose
across eight read heads recovered only 51.47% of the equal-L2 unrestricted
margin gain, whereas a signed-affine value mixture recovered 94.66%. Its
head-only and leave-one-out results also suggested a distributed, redundant
mechanism. This level asks two stricter questions:

1. what is the smallest read-head subset that preserves the signed mechanism;
2. can that label-aware Oracle be compiled into an input-conditioned reader
   whose inference does not use the target label?

## Frozen boundary

The trunk is the formally passed Level 6.18.3 seed707 checkpoint at 16 chunks.
The original Memory Probe and both Level 6.19 probes remain frozen. The failed
Level 6.18.9 candidate is excluded, optimizer/model search remains closed,
seed909 stays locked, and no protected test is opened.

Level 6.19.3 data are replayed only for the registered 255-subset calibration.
Router training, router validation, and the formal diagnostic use separate
fixed seeds. The formal diagnostic is opened once after all router checkpoints
have been selected by validation loss.

## Part A: all 255 head subsets

For each Memory-decodable source error, the script decomposes the full
signed-affine Oracle into eight head deltas. Every non-empty head subset is
summed and independently rematched to the registered per-example context L2
dose. All 255 conditions are replayed through the frozen nonlinear downstream
tail. The smallest subsets recovering at least 80% and 90% of the full signed
Oracle's actual deployed-margin gain are selected; ties use recovery and then
lexicographic head order. First-order gains remain a secondary geometry audit.
The 90% subset becomes the fixed router head mask.

This is still label-aware causal tomography. It does not select a deployable
model.

## Part B: exact target-simplex audit

On fresh formal primary examples, a two-start accelerated projected-gradient
solver projects the full signed-affine target delta onto the product of eight
32-slot attention simplices. This is a convex least-squares problem. The
solver has hard convergence gates for two-start delta agreement and projected
gradient mapping.

The audit answers whether the exact registered signed target is representable
by any valid per-head attention distribution. It does not claim to exhaust all
other equal-dose simplex directions.

## Part C: fixed label-free-inference routers

Three 8,930-parameter readers use the same query, pre-fusion state, source
context, attention map, projected value atoms, selected head mask, training
split, initialization protocol, optimizer, and dose ceiling:

- **signed router:** zero-mean signed slot coefficients in the frozen value
  basis;
- **non-negative router:** valid softmax attention followed by bounded source
  interpolation;
- **matched residual router:** the same scorer over a fixed unrestricted
  residual basis.

Training uses task labels and a fixed-weight distillation loss toward the
selected-subset signed Oracle. Inference uses no label, rival class, or Oracle
gradient. The signed router is tested against source, the two matched readers,
shuffled memory, rolled deltas, and head-permuted coefficients with Holm
correction across all six contrasts.

## Registered success gate

The signed reader passes only if all conditions hold on the fresh 4,096-sample
diagnostic:

- at least 25% of the full signed-Oracle deployed-margin gain is recovered on
  Memory-decodable source errors;
- all six signed specificity contrasts are positive after Holm correction;
- the lower 95% CI of full-panel accuracy change is no worse than -0.25%;
- all numerical, split, parameter-matching, and frozen-state integrity gates
  pass.

The 25% threshold is intentionally below the label-aware Oracle ceiling: this
is the first fixed input-conditioned compilation test, not another Oracle.

## Run

From `ist_v0_1`:

```powershell
python run_level6_19_4_local.py
```

The first completed formal run used 2,048 simplex iterations and is retained
under `formal/` as a numerical-audit failure.  Its worst two-start delta gap
was 0.01426 (gate 0.01), and its worst projected-gradient mapping was
2.31e-5 (gate 1e-5). The registered scientific target and tolerances were not
changed. The deterministic recovery only raises the convex-solver budget to
8,192 iterations. The script now defaults to the separate recovery directory;
the equivalent explicit command is:

```powershell
python run_level6_19_4_local.py --output experiments/level6_19_4/formal_recovery
```

The observed end-to-end recovery runtime on the RTX 5060 Laptop GPU was
**4 minutes 23 seconds**. Allow roughly **5-15 minutes** across laptop power
modes and competing GPU workloads. The script writes `progress.json`
throughout subset calibration, cache extraction, router training, formal
evaluation, and simplex projection.

Implementation smoke test:

```powershell
python run_level6_19_4_local.py --smoke-test --force
```

Smoke output uses separate seed ranges and is not scientific evidence. Formal
results are never overwritten unless `--force` is explicitly supplied.

## Formal artifacts and canonical result

`formal/` is the retained 2,048-iteration numerical-audit failure.
`formal_recovery/` is the canonical, integrity-passed 8,192-iteration result.
The subset calibration, router training, router checkpoint, labels, groups,
and all non-simplex diagnostic predictions are byte-for-byte or value-for-value
identical between the two runs. See `formal/NUMERICAL_AUDIT_FAILURE.md` and
`formal_recovery/ANALYSIS.md`.

The canonical recovery folder contains:

- `preregistration.json`;
- `subset_calibration.json`;
- `router_training.json` and `router_checkpoint.pt`;
- `result.json` and compact `summary.json`;
- per-example `predictions.json`;
- `minimal_heads_gated_read.png`;
- `ANALYSIS.md`, completed after the formal run.

All artifacts are stored under one experiment folder for GitHub publication
and independent reproduction.
