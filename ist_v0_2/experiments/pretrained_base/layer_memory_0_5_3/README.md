# Frozen Memory 0.5.3

Levels 0.5.1 and 0.5.2 wrote Fast Memory from Qwen's final layer but read it at
the input to decoder layer 20.  This stage removes that representation mismatch.

The adapter now installs a capture-only hook while processing the first chunk.
The complete layer-20 input sequence becomes the Fast Writer source.  The second
chunk reads those slots and injects them at the same layer-20 boundary.  Writes
still happen after each chunk's read, so chunk causality is unchanged.

All Level 0.5.2 losses and controls remain active: answer CE, layer representation
alignment, missing-context delta alignment, swapped-Memory margin, and held-out
zero/reset/swap interventions.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_3.py --dry-run
python run_pretrained_layer_memory_0_5_3.py --smoke-test --local-files-only --force
python run_pretrained_layer_memory_0_5_3.py --local-files-only
```
