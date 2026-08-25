# Frozen Memory 0.5.4

This is a read-only tomography stage over the best Level 0.5.3 checkpoints.  It
does not train, overwrite, or continue those models.

It measures Fast-slot geometry across held-out examples, effective rank, Writer
attention mass on the first-chunk fact span, Query-to-slot read entropy, and the
change in candidate logits caused by swapping complete Fast states between batch
examples.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_4.py --dry-run
python run_pretrained_layer_memory_0_5_4.py --smoke-test --local-files-only
python run_pretrained_layer_memory_0_5_4.py --local-files-only
```

The formal run uses 32 held-out examples for each of the three independent Level
0.5.3 checkpoints.  Results are written under this directory's `formal` folder.
