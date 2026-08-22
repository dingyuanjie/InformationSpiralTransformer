# Level 7.5.5.1

集中复验 Level 7.5.5 的 `L3 read/fusion` 最小充分候选，并以效果较稳定的 `L3 slot queries` 为对照。使用四个锁定的独立初始化和新的 screen/confirmation 数据种子。

```powershell
python run_level7_5_5_1_local.py --dry-run
python run_level7_5_5_1_local.py --smoke-test --force
python run_level7_5_5_1_local.py
```

正式实验包含 12 个新增训练分支，可断点续跑。
