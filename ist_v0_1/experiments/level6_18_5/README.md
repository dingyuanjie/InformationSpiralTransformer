# Level 6.18.5: surgical Memory-read routing rescue

Level 6.18.4 found 97.56% linearly decodable 16-chunk Memory, but only 92.97%
linearly decodable query-token state. The transferred Level 6.18.3 output head
already reaches 92.09% and is statistically indistinguishable from a refitted
linear query-hidden decoder. Level 6.18.5 tests whether the remaining deficit
can be repaired specifically at the final Memory-to-token read interface.

## Frozen source and trainable boundary

The source is the formally passed Level 6.18.3 checkpoint:

`experiments/level6_18_3/formal/rescued_head_checkpoint.pt`

Only six tensors in the final Spiral block are trainable:

- `blocks.2.memory_read.*`;
- `blocks.2.memory_fusion_gate.*`.

This is 24,896 parameters. Embeddings, both attention paths, every Memory
encoder/updater, lower layers, FFN/norms, and the successful output head remain
frozen. Training uses only 16-chunk query cross-entropy; there is no Probe loss.

The selected route acts after the final block has already produced its returned
Memory. Therefore it should change token behavior without changing persistent
Memory. The script verifies this architectural prediction numerically at every
chunk and layer for 8-, 12-, and 16-chunk trajectories.

## Stable validation gate

Training uses at most 500 optimizer updates, each accumulating four batch-2
micro-batches. Every 25 updates, fixed 8/12/16 screens are evaluated. Candidate
screens activate disjoint 256-example confirmation sets. A checkpoint must
reach at least 95% query accuracy at all three lengths on two successive
evaluations.

`routing_latest.pt`, `routing_best.pt`, and `routing_stable.pt` are separate.
The run is resumable from the latest 25-update checkpoint.

## Protected tests and success rule

After stable validation, the unchanged Level 6.18.3 model and routing-rescued
model are rerun on exactly the same 2,048 held-out examples at 8, 12, and 16
chunks. The 16-chunk change receives a paired bootstrap confidence interval and
McNemar test.

Formal success requires:

- stable 8/12/16 validation;
- protected 8-, 12-, and 16-chunk test accuracy each at least 95%;
- positive lower 95% CI for the 16-chunk improvement;
- only the six routing tensors changed;
- returned Memory states exactly invariant at every tested layer/chunk;
- rescued 16-chunk intact/reset/zero/batch-roll causal gate passed.

## Run

From the repository root:

```powershell
python run_level6_18_5_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 30–60 minutes,
depending on early stopping. If interrupted, run the same command to resume.
Do not use `--force` when resuming or after formal completion.

Artifacts are written to `experiments/level6_18_5/formal/`:

- `preregistration.json`;
- resumable latest/best/stable routing checkpoints;
- `result.json`, `summary.json`, and `predictions.json`;
- `routing_rescued_checkpoint.pt`;
- `routing_rescue.png`.

A pass causally confirms that the residual seed-707 16-chunk limitation was in
the final Memory-read route. It does not yet establish transfer to seed 909.

