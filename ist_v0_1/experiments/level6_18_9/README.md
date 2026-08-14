# Level 6.18.9: task-aligned Memory-read supervision

Level 6.18.8 confirmed that the update-500 read-context delta contains a small,
sample-specific task-aligned direction. Level 6.18.9 is one preregistered rescue
attempt that trains that boundary directly while suppressing the much larger
task-orthogonal drift.

## Source and trainable boundary

Training restarts from the formally passed Level 6.18.3 checkpoint, not from
the failed update-500 checkpoint.

Only four tensors in the final block's `memory_read` are trainable:

- `in_proj_weight` and `in_proj_bias`;
- `out_proj.weight` and `out_proj.bias`.

This is 16,640 parameters. The fusion gate, FFN, normalization, output head,
all Memory writers, embeddings, attention paths, lower blocks, and original
Memory Probe remain frozen.

## Single registered objective

Every optimizer update contains one batch at each of 8, 12, and 16 chunks. The
loss combines:

1. deployed correct-class margin loss;
2. a contrast requiring intact Memory margin to exceed batch-rolled-Memory
   margin;
3. a penalty on context drift orthogonal to the frozen deployed-margin
   gradient;
4. a small total context trust-region penalty relative to the source read.

There is one learning rate and at most 500 updates. This is not an optimizer or
hyperparameter search.

## Stable fail-closed gate

Every 25 updates, fixed 8/12/16 screens are evaluated. Candidate screens open a
disjoint 512-example confirmation panel. A checkpoint must reach at least 95%
query accuracy at all three lengths, improve 16-chunk continuous margin, and
retain 8/12 margin on two successive confirmations.

If this gate fails, protected tests and causal conditions remain unopened.
The latest checkpoint and complete validation trajectory are still saved for
mechanism analysis.

## Protected success rule

After a stable gate, source and rescued checkpoints are compared on identical
2,048-example protected tests at 8, 12, and 16 chunks. Formal success requires:

- at least 95% rescued accuracy at every length;
- positive 16-chunk paired accuracy and margin confidence intervals;
- significant 16-chunk paired margin sign-flip test;
- only the four Memory-read tensors changed;
- exact persistent-Memory invariance;
- passed 16-chunk intact/reset/zero/batch-roll causal gate.

Seed 909 remains locked.

## Run

From `ist_v0_1`:

```powershell
python run_level6_18_9_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 30-70 minutes.
Training is resumable from `read_supervision_latest.pt`: after an interruption,
run the same command without `--force`.

Artifacts are written to `experiments/level6_18_9/formal/`:

- `preregistration.json` and `baseline_validation.json`;
- resumable latest/best/stable checkpoints and progress JSON;
- `result.json`, `summary.json`, and `read_supervision.png`;
- protected predictions and `task_aligned_read_checkpoint.pt` only after the
  stable gate opens;
- the completed `ANALYSIS.md` should use `ANALYSIS_TEMPLATE.md`.
