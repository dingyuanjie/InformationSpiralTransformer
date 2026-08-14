# Level 6.19: frozen hard-example and residual-path diagnosis

Levels 6.18.8-6.18.9.1 established that a real task-aligned read direction
exists, but training the final `memory_read` mainly increases margin on already
correct examples and does not repair 16-chunk errors. Level 6.19 stops rescue
training and asks where correct-label information becomes inaccessible on hard
examples.

## Frozen source and disjoint splits

The model is the formally passed Level 6.18.3 seed707 checkpoint. The failed
Level 6.18.9 candidate is not used. All model parameters remain frozen.

Three new 16-chunk splits are disjoint:

- 2,048 examples for independent linear-decoder training;
- 512 examples for decoder validation;
- 4,096 examples for hard-example diagnosis.

The diagnostic hard group contains every deployed source error. Each error is
matched without replacement to a source-correct example with nearest top1-top2
confidence. At least 200 errors are required.

## Interfaces

Independent decoders measure correct-label information at:

1. all final-layer persistent-Memory slots;
2. pre-fusion Memory feature;
3. Memory-read context;
4. gate-times-context fusion delta;
5. fused FFN input;
6. isolated FFN output as a side branch;
7. pre-normalization residual sum;
8. query hidden state;
9. deployed logits.

The registered causal interpretation follows persistent Memory, read context,
fused feature, pre-norm residual, query hidden, and deployed behavior. The
isolated FFN output is never treated as the residual stream.

## Gradient and slot accessibility

For every diagnostic example, the script exactly reconstructs the frozen final
query computation and differentiates correct-versus-strongest-incorrect margin
with respect to read Memory, context, fused feature, and pre-norm residual.

The independent all-slot Memory decoder supplies a correct-label Memory-code
direction. The audit compares that direction with deployed Memory gradients,
per-slot gradient norms, and read attention. It also reports how much attention
and gradient mass reaches the four slots making the strongest decoder
contribution.

## Registered diagnosis

On source-error examples:

- Memory decoder below 75%: Memory encoding/composition failure;
- otherwise, the first registered-path decoder drop of at least 15 points
  identifies read, fusion, FFN/residual, or normalization failure;
- query-hidden decoder at least 75% with deployed failure identifies output
  alignment;
- otherwise the result is mixed.

This is a localization rule, not a rescue threshold.

## Run

From `ist_v0_1`:

```powershell
python run_level6_19_local.py
```

Expected runtime on an RTX 5060 Laptop GPU is approximately 25-60 minutes.
Artifacts are written to `experiments/level6_19/formal/`:

- `preregistration.json`;
- `result.json`, `summary.json`, and `predictions.json`;
- `linear_probes.pt`;
- `hard_example_diagnosis.png`;
- a completed `ANALYSIS.md` based on `ANALYSIS_TEMPLATE.md`.

This level does not train a model, use protected tests, or open seed909.
