# Frozen Memory 0.5.4.2

This is a locked independent confirmation of the directional Level 0.5.4.1
result.  It performs no method search and no training.

Pre-registered methods:

- `baseline`
- `prototype_center` (secondary)
- `prototype_pc1_topk4` (primary)

Calibration and held-out seed ranges are disjoint from Level 0.5.4.1.  Each of
the three checkpoints uses 64 new calibration examples and 256 new held-out
examples.  The primary method passes only if its normal accuracy has a Wilson
95% lower bound above 25% and paired two-sided tests are significantly positive
against both baseline and swapped Memory.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_4_2.py --dry-run
python run_pretrained_layer_memory_0_5_4_2.py --smoke-test --local-files-only
python run_pretrained_layer_memory_0_5_4_2.py --local-files-only
```
