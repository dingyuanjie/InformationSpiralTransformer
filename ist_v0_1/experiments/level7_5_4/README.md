# Level 7.5.4

目标：从 Level 7.5.3.4 的分布式 Memory 结果中寻找最小充分参数子集。

协议先行检查：

```powershell
python run_level7_5_4_local.py --dry-run
python run_level7_5_4_local.py --smoke-test
```

正式组合冻结训练沿用 7.5.3.4 的四个 outcome-stratified 来源、1000 steps、AdamW/RNG/data-stream 精确恢复协议。组合定义固定在 `preregistration.json`，不得追加未注册组合。
