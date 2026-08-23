# Level 7.6.4

从 Level 7.6 的 512-token checkpoint 接续至 1024、2048、4096，并在 8192 token 做冻结窗口评估。每 50 步保存恢复点；OOM 会被记录，不会自动改变 batch 或模型。

```powershell
python run_level7_6_4_local.py --dry-run
python run_level7_6_4_local.py
```

支持按模型、seed、stage 断点续跑。
