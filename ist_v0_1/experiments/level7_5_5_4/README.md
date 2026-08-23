# Level 7.5.5.4

L3 read/fusion 从第 1 步学习，L3 slot queries 在第 300 步后开放，并使用 10%、25%、50%、100% 梯度剂量。用于区分优化幅度冲突与结构性不兼容。

这里的“剂量”严格指反向传播得到的 slot gradient 乘数；AdamW 的锁定动量状态和权重衰减保持不变，因此结果解释为梯度剂量效应，而不是等价的 slot learning-rate 扫描。

```powershell
python run_level7_5_5_4_local.py --dry-run
python run_level7_5_5_4_local.py --smoke-test --force
python run_level7_5_5_4_local.py
```

正式实验包含 24 个新增训练分支，可断点续跑。
