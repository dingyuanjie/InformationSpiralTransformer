# Level 7.6.6.4：槽位几何与写入内容因果恢复

本阶段测试低表现seed的失败是否来自写入后槽位坍缩。对Memory更新门产生的槽位做保留原始范数、与原方向符号对齐的QR正交化，再按0.25、0.50或1.00剂量与原槽位插值。

条件包括L3剂量曲线、L1单层、L2+L3以及全层0.25。继续使用seed 313、1234作为低表现恢复组，seed 2026、7作为高表现副作用对照，并在32K近距与最远窗口使用完全相同的30个样本。除准确率外记录目标概率、目标logit margin和实际逐层槽位余弦冗余，避免遗漏未跨过argmax阈值的连续因果效应。

`SpiralMemory.slot_decorrelation_strength`默认是0，不改变正常模型行为。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_4_local.py --dry-run
python run_level7_6_6_4_local.py
```

每个seed、窗口、条件独立保存，可用同一命令续跑；不要加 `--force`。
