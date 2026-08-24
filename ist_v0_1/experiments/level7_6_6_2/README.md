# Level 7.6.6.2：低表现seed的定向因果恢复

本阶段以seed 313、1234为预注册低表现恢复组，以seed 2026、7为高表现副作用对照。在32K最近和最远窗口上，对第三层Memory传播比例实施0.50、0.32、0.20剂量，并测试L2+L3及全层0.32传播条件。

每个条件使用完全相同的30个样本，输出低表现组相对intact的配对恢复量、高表现组损害量、McNemar精确检验和五条件Holm校正。主终点是低表现组合并窗口准确率变化；只有在低组恢复且高组不受同等损害时，才支持传播过强是可选择的因果瓶颈。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_2_local.py --dry-run
python run_level7_6_6_2_local.py
```

每个seed、窗口、条件独立保存，可用同一命令续跑；不要加 `--force`。
