# Level 7.6.6.10：seed 2026双四槽组冗余—协同分析

本阶段冻结验证排名A组L3 `[31,3,5,12]` 与B组L3 `[28,4,13,24]`。主要检验族比较单独删除A、单独删除B、联合删除A+B与intact，并进行Holm校正。

额外条件测试仅保留A、仅保留B、分别增强A/B，以及删除一组后把另一组增强两倍。配对加性交互定义为 `AB - A - B + intact`；负值表示联合删除产生超过两个单组效应相加的超加性损害，并用10000次配对bootstrap给出95%区间。

32K近距和最远窗口各使用128个目标类别严格平衡样本。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_10_local.py --dry-run
python run_level7_6_6_10_local.py
```

每个窗口/条件独立保存，可用相同命令续跑；不要加 `--force`。
