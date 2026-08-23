# Level 7.6.5.1：8192 性能计时审计

本阶段不训练模型，只读取 Level 7.6.4 的 seed 313、4096-token checkpoint，解释 Level 7.6.4 与 Level 7.6.5 之间 Transformer 吞吐相差约 21 倍的原因。

审计同时记录首次冷启动、10 次预热后的 Python 墙钟与 CUDA Event 计时，并比较三种输入路径：固定样本回放、连续 RNG 生成、逐样本重设 seed。每项测量 30 次前向、独立重复 3 次，同时保存 CUDA/SDP/确定性配置和峰值显存。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_5_1_local.py --dry-run
python run_level7_6_5_1_local.py
```

每个“模型 × 输入模式 × 重复”完成后都会保存，意外中断后执行相同命令即可续跑。不要加 `--force`。
