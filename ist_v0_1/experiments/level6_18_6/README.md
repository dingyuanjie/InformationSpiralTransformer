# Level 6.18.6: frozen final-block routing tomography

Level 6.18.5 completed all 500 optimizer updates but failed its stable validation
gate. The final-block `memory_read` and `memory_fusion_gate` tensors changed,
while 16-chunk validation barely moved. Level 6.18.6 is a frozen comparison
that locates where that update changed the representation and where useful
information was lost.

## Compared checkpoints

- source: `experiments/level6_18_3/formal/rescued_head_checkpoint.pt`;
- update 500: `experiments/level6_18_5/formal/routing_latest.pt`.

Before evaluation, the script requires the checkpoints to differ in exactly
the six final-block routing tensors (24,896 parameters). It rejects any other
parameter change. No model or original Probe parameter is updated in this
experiment.

## Frozen interfaces

At the final query position, independently refitted standardized linear probes
measure:

1. final-layer persistent Memory, all slots concatenated;
2. the Memory module's pre-fusion token feature;
3. `memory_read` context;
4. fusion-gate output;
5. the fused feature entering the FFN;
6. FFN output;
7. final `norm2` query hidden state;
8. the deployed head's 16 logits.

The deployed argmax is reported separately. Persistent Memory and the
pre-fusion feature must be exactly equal across models. All probes use disjoint
train/validation/test sets, and both checkpoints see exactly the same examples.
Probe changes and deployed behavior receive paired bootstrap intervals and
McNemar tests. Source and update-500 Probe fits at a given interface use the
same random seed, so an invariant representation produces an identical fitted
decoder instead of a spurious optimizer-seed difference.

The experiment repeats the tomography at 8, 12, and 16 chunks. At 16 chunks it
also reruns intact/reset/zero/batch-roll Memory interventions as diagnostics,
not as a recovery gate.

## Run

From `ist_v0_1`:

```powershell
python run_level6_18_6_local.py
```

Expected time on an RTX 5060 Laptop GPU is roughly 25-55 minutes. Completed
lengths are stored independently and reused after an interruption. Use the same
command to resume. `--force` intentionally reruns every length.

Outputs are written to `experiments/level6_18_6/formal/`:

- `preregistration.json`;
- `chunks8.json`, `chunks12.json`, and `chunks16.json`;
- per-length prediction files plus consolidated `predictions.json`;
- `result.json` and `summary.json`;
- `routing_tomography.png`;
- the completed `ANALYSIS.md` should be based on `ANALYSIS_TEMPLATE.md`.

## Frozen 16-chunk interpretation rule

- read-context and query-hidden improvements both below 1 point:
  `no_generalizable_route_change`;
- context improves by at least 2 points but query hidden improves below 1:
  `post_context_erasure`;
- query hidden improves by at least 2 points but deployed behavior improves
  below 1: `output_mismatch`;
- deployed behavior improves by at least 2 points: `route_generalized`;
- otherwise: `mixed_or_small_effect`.

This level does not authorize a broader intervention. Its result selects the
next boundary: read supervision, fusion plus FFN/norm, or output alignment.
