# Level 7.5.5.5

在第 300 步开放 L3 slot queries，比较保留 AdamW 状态、清零一阶动量、同时清零一阶和二阶动量。分别使用 25% 与 100% slot 梯度剂量。

```powershell
python run_level7_5_5_5_local.py --dry-run
python run_level7_5_5_5_local.py --smoke-test --force
python run_level7_5_5_5_local.py
```

正式实验包含 32 个新增训练分支，可断点续跑。weight decay 未改变，因此本阶段只解释 AdamW moment-state 的因果作用。
