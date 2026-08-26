# Frozen Memory 0.5.4.1

This stage performs read-only interventions on Level 0.5.3 checkpoints.  A
separate validation calibration set estimates each checkpoint's Fast-Memory
prototype and first centered principal component.  Held-out evaluation then
compares baseline, prototype removal, within-example slot centering, PC1 removal,
temperature sharpening, top-4 reading, and combined interventions.

Every method is evaluated under both matched and cross-example swapped Memory.
No checkpoint is trained or overwritten.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_4_1.py --dry-run
python run_pretrained_layer_memory_0_5_4_1.py --smoke-test --local-files-only
python run_pretrained_layer_memory_0_5_4_1.py --local-files-only
```
