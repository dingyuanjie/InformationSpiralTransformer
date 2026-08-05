# InformationSpiralTransformer

## v0.1 training experiment

`ist_v0_1/train_compare.py` trains Information Spiral Transformer v0.1 and a
standard Transformer under the same configuration. The built-in benchmark is a
deterministic recursive-sequence masked-token recovery task, so it needs no
external dataset download.

```bash
cd ist_v0_1
python train_compare.py
```

For a quick smoke test:

```bash
python train_compare.py --epochs 1 --train-samples 64 --validation-samples 32
```

The script prints training/validation loss, masked-token accuracy, parameter
count and elapsed time. It also writes the complete result to
`comparison_results.json` (override with `--output`). This experiment is an
engineering baseline rather than evidence that either architecture is
universally superior; reliable conclusions require repeated seeds and real
datasets.

## Long-context test

The long-context benchmark trains both models to retrieve the first token at a
masked final position using lengths 32 and 64, then tests extrapolation through
length 512. It records accuracy, batch latency, throughput and peak CUDA memory.

```bash
cd ist_v0_1
python long_context_test.py
```

Use `--test-lengths 32 64 128 256 512 1024` to extend the context range. Results
are saved to `long_context_results.json`.

## Memory visualization

Train a small IST model and inspect what Spiral Memory retains:

```bash
cd ist_v0_1
python visualize_memory.py
```

The generated `memory_visualization.png` shows compression attention over input
positions, per-slot activation, update-gate behavior and similarity between
memory slots. Numeric diagnostics are also saved to `memory_visualization.json`.

## v0.2 memory specialization

Spiral Memory now uses a learnable query for each memory slot, a slot-specific
update gate and an auxiliary diversity loss. Training scripts apply this loss
with `--diversity-weight 0.1` by default. Set it to `0` for a diversity-loss
ablation. Memory
diagnostics additionally report attention entropy, effective context tokens and
diversity loss.

## v0.3 ablation

Run the position-encoding and memory-diversity sweep:

```bash
cd ist_v0_1
python ablation_test.py
```

The sweep compares RoPE and learned absolute positions across diversity weights
`0`, `0.001`, `0.01`, `0.05` and `0.1`, then writes the ranked results to
`ablation_results.json`.

## v0.4 position milestone

Run the five-seed comparison of Absolute, Sinusoidal, RoPE and Scaled RoPE:

```bash
cd ist_v0_1
python position_ablation.py
```

The report includes mean, standard deviation and worst-seed accuracy in
`position_results.json`.
