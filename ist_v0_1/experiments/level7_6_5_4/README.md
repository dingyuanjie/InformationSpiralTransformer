# Level 7.6.5.4：无分配污染的平衡长度缩放复验

本阶段重新测量 1024、2048、4096、8192 的速度交叉点。所有模型与输入在计时前完成分配和预热；整个延迟测量阶段不调用 `empty_cache()`。12 个“模型 × 长度”条件先随机排列，再循环移位形成 12 轮日程，因此每个条件在 12 个测试位置恰好出现一次。

每个 block 连续前向 30 次并同时使用墙钟和 CUDA Event。峰值显存在计时完成后用独立探针测量，报告总峰值及扣除常驻模型/输入后的增量峰值，避免显存管理操作污染延迟。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_5_4_local.py --dry-run
python run_level7_6_5_4_local.py
```

共 144 个计时 block，每个 block 独立保存。中断后执行相同命令续跑，不要加 `--force`。
