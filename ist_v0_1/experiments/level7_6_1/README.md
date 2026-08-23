# Level 7.6.1

锁定 Level 7.6 的 15 个最终 checkpoint，每个 seed、每个长度评估 1000 个相同样本，生成 Wilson 95% 区间和相对参数匹配 Transformer 的逐样本配对差值区间。

```powershell
python run_level7_6_1_local.py --dry-run
python run_level7_6_1_local.py
```

这是纯评估实验，不会继续训练，支持按模型/seed 断点续跑。
