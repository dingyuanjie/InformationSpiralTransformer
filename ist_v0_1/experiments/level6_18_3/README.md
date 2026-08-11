# Level 6.18.3: surgical output-head rescue

Level 6.18.2 found that the failed seed-707 12-chunk checkpoint has 98.63%
linearly decodable Memory and 96.97% linearly decodable final query-token state,
while its deployed task head reaches only 90.33%. Level 6.18.3 tests the causal
prediction that changing only the output head can expose this latent behavior.

## Frozen source and mutation boundary

The source is the same diagnostic checkpoint used by Level 6.18.2:

`experiments/level6_18_1/formal/seed707/transition_8_to_16_bridge_best.pt`

Every IST parameter is frozen. A 16-class linear head is initialized to exactly
reproduce rows 0–15 of `model.output`, trained on frozen 12-chunk query-hidden
features, and converted back to raw hidden-state coordinates. Rows 16–18 remain
bit-identical. The script audits every model tensor and fails the mutation gate
if any non-output tensor changes.

## Validation-only rescue-dose selection

The fully fitted head is interpolated with the original head:

`head(alpha) = (1 - alpha) * original + alpha * fitted`

The frozen candidate grid is `alpha = 0.0, 0.1, ..., 1.0`. Candidates are
evaluated on independent 8- and 12-chunk validation sets. A candidate is
eligible only when both accuracies are at least 95%. Among eligible candidates,
selection maximizes 12-chunk accuracy, then 8-chunk accuracy, then prefers the
smaller update. No 16-chunk example participates in training or selection.

## Protected paired tests

The untouched and selected heads are rerun on exactly the same examples at 8,
12, and 16 chunks. The 16-chunk result is protected exploratory transfer and is
not a primary success gate. The 12-chunk change is reported with a paired
bootstrap confidence interval and McNemar test.

Primary success requires all of:

- validation selection finds an eligible dose;
- only the output head changes;
- protected 8-chunk accuracy remains at least 95%;
- protected 12-chunk accuracy reaches at least 95%;
- the 12-chunk paired improvement 95% CI has lower bound above zero;
- rescued 12-chunk Memory causality passes.

## Causal preservation

On a separate protected 12-chunk dataset, the rescued model is evaluated under
intact, reset, zero, and batch-rolled Memory. The gate requires intact accuracy
at least 95%, every disrupted condition at most 20%, and local accuracy at least
90%.

## Run

From the repository root:

```powershell
python run_level6_18_3_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 15–35 minutes.
Completed formal results are reused; do not use `--force` unless intentionally
replacing the run.

Artifacts are written to `experiments/level6_18_3/formal/`:

- `preregistration.json`: frozen selection and success protocol;
- `result.json` and `summary.json`: full and compact results;
- `predictions.json`: protected paired and causal predictions;
- `rescued_head_checkpoint.pt`: rescued model plus original diagnostic probe;
- `head_only_rescue.png`: GitHub-ready result figure.

Passing this level would causally confirm a head-alignment bottleneck for this
seed/checkpoint. It would not yet establish cross-initialization recovery or
retroactively pass Level 6.18.1.

