# Frozen Memory 0.5.1

This is the held-out generalization successor to the fixed-32 Level 0.5 gate.
Every optimizer step receives new 1024-token examples.  The frozen Qwen backbone
processes two 512-token chunks and historical Fast Memory is injected before
decoder layer 20 (the fourth-from-last block in the 24-layer 0.5B model).

Training combines the normal answer loss with a cross-example Memory-swap margin.
Same-label swaps are excluded from the margin because this four-choice task cannot
observe their semantic mismatch at the output label.

The selected checkpoint maximizes the conservative validation score:

`normal - max(zero_fast, reset_memory, swap_fast)`

Formal success requires held-out normal accuracy above chance by Wilson 95% lower
bound and significant positive paired effects against all three controls.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_1.py --dry-run
python run_pretrained_layer_memory_0_5_1.py --smoke-test --local-files-only --force
python run_pretrained_layer_memory_0_5_1.py --local-files-only
```
