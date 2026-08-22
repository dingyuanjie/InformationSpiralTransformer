# Level 7.5.5.3

比较 read-only、从第 0 步联合训练，以及在 300/600 步后开放 L3 slot queries，判断负交互来自同步早期更新还是长期表示不兼容。

```powershell
python run_level7_5_5_3_local.py --dry-run
python run_level7_5_5_3_local.py --smoke-test --force
python run_level7_5_5_3_local.py
```

正式实验包含 20 个新增训练分支，可断点续跑。
