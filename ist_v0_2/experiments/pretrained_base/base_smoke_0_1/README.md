# Base Smoke 0.1 — Identity-Preserving Adapter

The frozen Qwen branch uses `hidden + tanh(scale) * (memory_feature - hidden)`, with scale initialized to exactly zero. Step-zero logits must exactly equal Base logits. Training adds final-token KL distillation to preserve language behavior.

```powershell
python run_pretrained_base_smoke_0_1.py --dry-run
python run_pretrained_base_smoke_0_1.py
```
