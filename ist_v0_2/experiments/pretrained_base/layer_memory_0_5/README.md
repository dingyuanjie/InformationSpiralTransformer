# Frozen Memory 0.5

This stage tests whether historical Fast Memory becomes learnable when it is
injected inside frozen Qwen rather than after every Transformer block.

The first 512-token chunk writes Fast Memory.  The second chunk reads those
slots before Qwen's fourth-from-last decoder block, leaving four frozen blocks
to fuse the retrieved information with the query.  The current chunk is written
only after its forward pass, so the read path is causally historical.

This is deliberately a fixed 32-example overfit gate.  It must pass before any
unique-stream generalization run is justified.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5.py --dry-run
python run_pretrained_layer_memory_0_5.py --smoke-test --local-files-only --force
python run_pretrained_layer_memory_0_5.py --local-files-only
```

Pass criteria: normal accuracy at least 95%, a drop of at least 50 percentage
points under both `zero_fast` and `reset_memory`, sustained for two evaluations.
`swap_fast` and `roll_fast` are reported as additional binding controls.
