# Level 7.5.3.4

冻结状态下对 Memory pathway 做细粒度因果干预：分别冻结 L2/L3 的 `slot_queries`、写入核心（encoder/key/attention）和读取融合（memory_read/fusion_gate）。每个干预都从同一来源 checkpoint 恢复模型、Probe、AdamW、CPU/CUDA RNG，并保持 update gate 活跃。

正式运行：

```powershell
python run_level7_5_3_4_local.py
```

预计 RTX 5060 约 6–8 小时。可用 `--smoke-test --force` 做协议与冻结审计，`--dry-run` 只检查锁定协议。结果、checkpoint、轨迹面板和 `ANALYSIS.md` 均写入本目录，便于上传 GitHub 复现。
