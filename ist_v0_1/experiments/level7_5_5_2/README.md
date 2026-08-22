# Level 7.5.5.2

检验 L3 slot queries 是否能够稳定强效但初始化敏感的 L3 read/fusion 恢复路径，并报告逐初始化联合增益与加性交互项。

```powershell
python run_level7_5_5_2_local.py --dry-run
python run_level7_5_5_2_local.py --smoke-test --force
python run_level7_5_5_2_local.py
```

正式实验包含 16 个新增训练分支，可断点续跑。
