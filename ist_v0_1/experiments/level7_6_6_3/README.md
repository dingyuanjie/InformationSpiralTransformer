# Level 7.6.6.3：第三层读取—融合定向恢复

本阶段继续以seed 313、1234为低表现恢复组，以seed 2026、7为高表现副作用对照。传播比例保持原始状态，只干预第三层Memory读取集中度和融合强度。

读取条件基于原始注意力显著性选择top-24或top-16槽位后重新执行Memory读取；融合条件把第三层融合门下限设为0.35或高表现组附近的0.50。联合条件测试top-16与两个融合剂量。每个条件使用相同的30个32K样本，输出实际读取熵、融合门、配对恢复量、McNemar精确检验及Holm校正。

`SpiralBlock`中的两个控制默认均为关闭状态，不改变正常模型行为。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_3_local.py --dry-run
python run_level7_6_6_3_local.py
```

每个seed、窗口、条件独立保存，可用同一命令续跑；不要加 `--force`。
