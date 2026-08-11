# Level 6.18.7: bidirectional activation transplant

Level 6.18.6 showed that the Level 6.18.5 update first changes the final block's
read context. At 16 chunks, read-context linear decoding increased by 1.37
points, but fused-feature, query-hidden, and deployed behavior did not retain a
confirmed gain. Level 6.18.7 tests the causal role of that changed activation
without training any parameter.

## Frozen models

- source: `experiments/level6_18_3/formal/rescued_head_checkpoint.pt`;
- update 500: `experiments/level6_18_5/formal/routing_latest.pt`.

The inherited checkpoint audit requires exactly six changed tensors and 24,896
changed parameters, all under the final `memory_read` and `memory_fusion_gate`.

## Transplants

On identical examples, only the last query position of the final chunk is
replaced. Transplants run in both directions:

- update-500 activation into the source receiver;
- source activation into the update-500 receiver.

The panel replaces:

1. read context only;
2. gate activation only;
3. read context plus gate;
4. fused feature entering the FFN;
5. FFN output;
6. final query hidden state.

The context-only forward condition is the primary mechanism test: the updated
context is passed through the source gate and all unchanged downstream layers.
Comparing it with the full update measures the effect of switching to the
updated gate after the updated context. Gate-activation-only is a deliberately
synthetic side diagnostic and is not interpreted as a clean factorial main
effect.

Because route outputs do not enter returned persistent Memory, all prefix
chunks are computed once. The script asserts exact returned-Memory invariance.
It also requires context+gate, fused-feature, FFN-output, and query-hidden
patches to reproduce donor logits and predictions exactly in both directions.

## Statistics and decision

At 16 chunks, three primary exact paired McNemar tests form one Holm-corrected
family:

- source to updated-context-through-source-gate;
- that context condition to the full update;
- source to full update.

The reverse direction is a corroborative restoration panel. Paired bootstrap
intervals are also reported for effect size.

## Run

From `ist_v0_1`:

```powershell
python run_level6_18_7_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 10-25 minutes.
Completed 8-, 12-, and 16-chunk panels are independently resumable. Run the
same command after an interruption; do not add `--force` when resuming.

Formal artifacts are written to `experiments/level6_18_7/formal/`:

- `preregistration.json`;
- per-length result and prediction files;
- `result.json`, `summary.json`, and `predictions.json`;
- `activation_transplant.png`;
- the completed `ANALYSIS.md` based on `ANALYSIS_TEMPLATE.md`.

A transplant result selects a causal boundary. It does not authorize an
optimizer search or establish transfer to seed 909.
