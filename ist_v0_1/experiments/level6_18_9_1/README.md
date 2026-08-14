# Level 6.18.9.1: frozen validation-calibration audit

Level 6.18.9 completed all 500 updates but never opened confirmation because its
128-example 12-chunk screen stayed at 120/128 = 93.75%. The absolute 0.94 screen
threshold requires 121/128. Meanwhile, 16-chunk screen accuracy and continuous
margin improved. Level 6.18.9.1 determines whether this was small-panel gate
granularity or a genuine lack of cross-length retention.

## Frozen checkpoints

- source: the formally passed Level 6.18.3 checkpoint;
- candidate: `experiments/level6_18_9/formal/read_supervision_latest.pt` at
  update 500.

No parameter is updated. The audit requires exactly four changed final
`memory_read` tensors (16,640 parameters), an unchanged original Probe, and
bitwise-identical persistent Memory.

## Two independent validation panels

Two new, disjoint panels each contain 2,048 examples at 8, 12, and 16 chunks.
Both panels must pass independently; results are not rescued by aggregation.

At 8 and 12 chunks, the candidate must be non-inferior to source:

- paired accuracy-change 95% CI lower bound at least -1 point;
- paired margin-change 95% CI lower bound at least -0.02;
- paired cross-entropy-change 95% CI upper bound at most +0.01.

At 16 chunks, each panel must additionally reach at least 95% absolute
accuracy and show positive, significant continuous-margin superiority.

These tests use new validation seeds. The existing protected tests and seed909
remain locked regardless of the result.

## Run

From `ist_v0_1`:

```powershell
python run_level6_18_9_1_local.py
```

Expected time on an RTX 5060 Laptop GPU is approximately 8-20 minutes.
Artifacts are written to `experiments/level6_18_9_1/formal/`:

- `preregistration.json`;
- `panel_a.json`, `panel_b.json`, and per-panel predictions;
- `result.json`, `summary.json`, and `predictions.json`;
- `validation_calibration.png`;
- a completed `ANALYSIS.md` based on `ANALYSIS_TEMPLATE.md`.

A pass authorizes only a separately registered decision about opening protected
tests. It does not itself open them or authorize more training.
