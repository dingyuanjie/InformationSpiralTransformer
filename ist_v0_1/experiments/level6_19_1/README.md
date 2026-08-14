# Level 6.19.1: selective slot-read causal intervention

Level 6.19 found that the frozen 16-chunk source model retains linearly
decodable correct-label information in persistent Memory on 81.95% of its
errors, but loses most of it when Memory is read into the final query context.
This level tests whether that boundary is causal.

## Frozen boundary

The source is the formally passed Level 6.18.3 seed707 checkpoint. The failed
Level 6.18.9 candidate is not used. All model parameters, the original Memory
Probe, and the independent Level 6.19 probes remain frozen. Persistent Memory
must remain exactly invariant under every intervention. Seed909 and protected
tests remain locked.

The Level 6.19 all-slot Memory Probe defines a label-aware correct-versus-
deployed-rival contribution for every slot. This is a causal mechanism test,
not a deployable inference method, because the selection uses the true label.

## Intervention and controls

Only the final block's final-query `memory_read` attention logits are changed.
Adding `log(odds)` to four selected slots preserves attention normalization and
multiplies their pre-normalization attention odds.

- main condition: top-four Memory-code slots at 4x odds;
- dose curve: the same slots at 2x, 4x, and 8x odds;
- equal-dose controls: bottom-four contribution slots, another example's
  top-four slots, and four independent random four-slot selections;
- no-op control: the unmodified frozen source;
- positive control: move final-query context along its exact frozen
  correct-versus-rival gradient, L2 matched to the top-four 4x context change.

The confirmatory population is source-error examples for which the frozen
Memory Probe is correct. Context-margin specificity and deployed-correction
specificity each form a separate four-comparison Holm family. Full-panel and
confidence-matched-correct retention are also required.

## Run

From `ist_v0_1`:

```powershell
python run_level6_19_1_local.py
```

The script writes progress to `experiments/level6_19_1/formal/progress.json`.
Expected runtime on an RTX 5060 Laptop GPU is approximately **5-12 minutes**.
The main cost is 13 frozen final-query forwards plus one differentiable forward
per batch; it does not train a model.

Outputs:

- `preregistration.json`;
- `result.json` and `summary.json`;
- `predictions.json` with per-example conditions and selected slots;
- `selective_slot_read_intervention.png`;
- `ANALYSIS.md`, completed after the formal run.

Do not add `--force` unless intentionally replacing a completed formal result.
`--smoke-test` only relaxes the fixed formal sample-size guard for code checks
and automatically moves to a disjoint smoke-only seed range; its output is not
scientific evidence.
