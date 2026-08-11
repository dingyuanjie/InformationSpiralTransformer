# Level 6.18.2: frozen 12-chunk Memory–Query decoupling

Level 6.18.1 recovered seed 707 from 4 to 8 chunks, but its 8-to-12 bridge
stalled at 80–86.25% task accuracy while the original Memory probe remained as
high as 92.5%. This experiment determines whether the 12-chunk bottleneck is in
Memory encoding, Memory-to-token routing, or the final output head.

This is a post-hoc mechanism diagnosis. It does not change the failed Level
6.18.1 result and is not a new recovery-success test.

## Frozen source

The only model checkpoint is:

`experiments/level6_18_1/formal/seed707/transition_8_to_16_bridge_best.pt`

All IST and original-probe parameters are frozen. The script verifies that the
checkpoint metadata identifies the 12-chunk bridge. Only new diagnostic linear
probes receive gradients.

## Disjoint datasets

- diagnostic-probe train: 2,048 examples;
- validation/early stopping: 512 examples;
- final test: 1,024 examples.

The three splits use fixed disjoint seeds. Accuracy and sample-level prediction
overlap are reported only on the held-out test split.

## Diagnostic features

Seven independently standardized linear probes are fit:

- mean of the 32 third-layer Memory slots;
- concatenation of all 32 third-layer slots;
- concatenated per-layer Memory means;
- concatenation of all layers and slots;
- final query-token hidden state immediately before the model output layer;
- third-layer Memory concatenated with the query hidden state;
- all Memory concatenated with the query hidden state.

Because the trained task output is itself linear in the query-token hidden
state, a successful refitted query-hidden probe specifically diagnoses output
head misalignment. A successful Memory probe with a weak query-hidden probe
instead diagnoses failure between persistent Memory and the query token.

## Frozen decision rule

- best Memory test accuracy below 90%: Memory propagation/encoding bottleneck;
- Memory and query-hidden probes at least 95%, with a weak original task head:
  output-head alignment bottleneck;
- Memory at least 95% and at least 5 points above query hidden:
  Memory-to-query-token routing bottleneck;
- otherwise: mixed or ambiguous.

The script additionally reports `probe_only_correct`: test examples for which a
refitted probe recovers the target while the original task head is wrong.

## Causal control

The same 1,024 held-out examples are evaluated under intact, reset, zero, and
batch-rolled Memory. Causal labels are asserted to be identical across all four
conditions. This distinguishes an accessible long-range signal from a local or
dataset artifact.

## Run

From the repository root:

```powershell
python run_level6_18_2_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is roughly 10–25 minutes. Completed
results are reused on restart; use `--force` only when intentionally replacing
the diagnostic result.

Formal artifacts are written to `experiments/level6_18_2/formal/`:

- `preregistration.json`: complete frozen protocol;
- `result.json`: checkpoint, behavior, probes, overlap, causal results, diagnosis;
- `summary.json`: compact machine-readable conclusion;
- `predictions.json`: held-out labels and per-method predictions;
- `memory_query_decoupling.png`: GitHub-ready accuracy comparison.

