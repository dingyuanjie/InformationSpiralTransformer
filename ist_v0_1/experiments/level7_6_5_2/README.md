# Level 7.6.5.2：1024–8192 长度缩放曲线审计

本阶段固定 Level 7.6.4 的 seed 313、4096-token checkpoint，以 batch=1 和相同 BF16 推理协议，在 1024、2048、4096、8192 四个长度上比较参数匹配 Transformer、IST-full 和 IST-stable。

每个点预热 10 次并独立重复 3 次；根据长度分别测量 120、80、50、30 次前向。输出稳态延迟、吞吐、CUDA 峰值显存、重复变异系数，以及延迟和显存相对上下文长度的 log-log 经验缩放指数。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_5_2_local.py --dry-run
python run_level7_6_5_2_local.py
```

每个“模型 × 长度 × 重复”完成后立即保存；中断后执行同一命令续跑，不要加 `--force`。
