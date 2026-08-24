# Level 7.6.6.1：32K成功/失败 seed 的 Memory 状态对照

本阶段只分析 IST-full。根据 Level 7.6.6 预注册 seed 2026、7为高表现组，seed 313、1234为低表现组，seed 42作为中间参照。在32K最近与最远距离窗口各采集20个样本。

逐层、逐槽记录写入熵、更新门、槽位范数与余弦冗余、Memory读取分布、传播读取分布、融合门和传播比例。输出高/低组均值差和标准化差异，用来定位32K外推稳定性最可能对应的内部状态；本阶段是诊断性对照，不把相关性直接解释为因果机制。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_1_local.py --dry-run
python run_level7_6_6_1_local.py
```

每个 seed/窗口独立保存，可用同一命令续跑；不要加 `--force`。
