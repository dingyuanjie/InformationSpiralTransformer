# Base Smoke 0.3.2 — Separate Optimization

Continue directly from the 0.3.1 step-1000 checkpoint. Fast Memory uses LR 8e-5 and independent norm clipping; the FP32 residual gate uses LR 2e-6 and gradient clamp 5. No Qwen, Slow, Episodic, or Router parameters are opened.

```powershell
python run_pretrained_base_smoke_0_3_2.py --dry-run
python run_pretrained_base_smoke_0_3_2.py --local-files-only
```
