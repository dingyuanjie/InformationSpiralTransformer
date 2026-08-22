# Level 7.5.5

从全 Memory 冻结基线出发，仅开放预注册的单组件、整层或跨层同功能组件继续学习，寻找最小充分恢复子集。

```powershell
python run_level7_5_5_local.py --dry-run
python run_level7_5_5_local.py --smoke-test --force
python run_level7_5_5_local.py
```

正式实验包含 4 个固定来源和 12 个恢复分支，共 48 个新增训练分支，可断点续跑。
