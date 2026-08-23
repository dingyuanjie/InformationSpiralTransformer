# Level 7.5.5.6

固定 25% slot 梯度与一阶动量清零，进一步拆分 AdamW step/bias correction 和 decoupled weight decay。二阶动量始终保留。

```powershell
python run_level7_5_5_6_local.py --dry-run
python run_level7_5_5_6_local.py --smoke-test --force
python run_level7_5_5_6_local.py
```

正式实验包含 28 个新增训练分支，可断点续跑。
