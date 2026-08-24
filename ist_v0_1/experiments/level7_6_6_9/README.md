# Level 7.6.6.9：seed 2026分布式冗余剂量曲线

本阶段按 Level 7.6.6.5 的验证集Probe准确率对seed 2026全部96个“层×槽”排序，构造嵌套top-4、8、12、16、24删除曲线。两条独立固定随机排列提供相同剂量的随机删除基线。

32K近距和最远窗口各使用96个类别严格平衡样本。五个验证排名剂量与intact的配对检验构成主要检验族，并使用Holm校正。崩溃阈值定义为合并窗口上首次出现负向且Holm显著效应的最小删除剂量。排名删除与两条随机曲线也分别进行配对比较。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_9_local.py --dry-run
python run_level7_6_6_9_local.py
```

每个窗口/条件独立保存，可用相同命令续跑；不要加 `--force`。
