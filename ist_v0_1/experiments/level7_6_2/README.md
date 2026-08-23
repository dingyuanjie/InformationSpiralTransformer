# Level 7.6.2

使用冻结的 Level 7.6 checkpoint 测试四种语义不变的 held-out 分布：needle 位于早期、中部、后期，以及同一目标跨四段重复。每个任务/长度合计 1250 个配对样本。

```powershell
python run_level7_6_2_local.py --dry-run
python run_level7_6_2_local.py
```

纯评估任务，按模型和 seed 断点续跑。严格参数量重新训练不属于本阶段，当前继续使用 Level 7.6 的近似参数匹配 checkpoint。
