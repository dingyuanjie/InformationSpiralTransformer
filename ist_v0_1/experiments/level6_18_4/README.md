# Level 6.18.4: residual 16-chunk decoupling

Level 6.18.3 changed only the output head and improved the untouched seed-707
checkpoint from 83.25% to 91.31% at 16 chunks, although no 16-chunk example was
used for training or head selection. Level 6.18.4 locates the remaining gap to
the 95% formation target.

This is a frozen post-hoc diagnosis. It performs no IST, Probe, or output-head
updates.

## Shared backbone and two deployed heads

The script loads:

- the untouched Level 6.18.1 diagnostic-best checkpoint;
- the formally passed Level 6.18.3 rescued-head checkpoint.

It verifies that every non-output tensor, the diagnostic Probe, and output rows
16–18 are bit-identical. The untouched backbone is run once, and both the
original and transferred 12-chunk rescue heads read the same hidden states.

## Frozen tomography

Disjoint 16-chunk datasets contain 2,048 training, 512 validation, and 1,024
held-out test examples. Seven standardized linear probes are refit on:

- third-layer Memory mean;
- all third-layer slots;
- concatenated per-layer Memory means;
- all layers and slots;
- final query-token hidden state;
- third-layer Memory plus query hidden;
- all Memory plus query hidden.

The held-out test also reports sample overlap among the original head,
transferred rescue head, query-hidden decoder, and all-Memory decoder.

## Frozen diagnosis rule

- best Memory accuracy below 90%: Memory propagation/encoding degradation;
- Memory and query-hidden accuracy at least 95%, but transferred head below
  95%: residual length-specific output alignment;
- Memory at least 95% and at least 5 points above query hidden:
  Memory-to-query-token routing degradation;
- transferred head at least 95%: no residual 16-chunk deficit;
- otherwise: mixed or ambiguous.

## Shared-trajectory causal control

Original and transferred heads are evaluated on the same 1,024 examples under
intact, reset, zero, and batch-rolled Memory. Both heads read each trajectory in
the same forward pass. Labels are asserted identical across conditions. The
diagnostic causal gate requires intact accuracy at least 80%, every disrupted
accuracy at most 20%, and local accuracy at least 90%.

## Run

From the repository root:

```powershell
python run_level6_18_4_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 10–30 minutes.
Completed results are reused; do not use `--force` unless intentionally
replacing the formal diagnosis.

Artifacts are written under `experiments/level6_18_4/formal/`:

- `preregistration.json`;
- `result.json` and `summary.json`;
- `predictions.json` with held-out and causal per-sample predictions;
- `residual_16_decoupling.png`.

This level decides what kind of intervention is justified next. It does not
open seed 909 and cannot retroactively change Level 6.18.1.

