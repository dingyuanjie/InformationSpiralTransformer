# Level 7.6.6.5：32K Memory内容可解码性审计

本阶段不干预模型。每个seed在32K近距和最远窗口各采集160个类别严格平衡样本，沿正常推理路径保存三层32×64完整Memory状态。

每个类别固定划分12个训练、4个验证、4个测试样本。测试集不参与槽位选择、模型选择或早停。Probe包括96个逐槽岭线性Probe、3个逐层拼接线性Probe、全层拼接线性Probe及128隐藏单元的小型MLP。最佳单槽只能由验证集选择，再报告隔离测试集准确率。

目标是区分：低seed的Memory中根本没有目标信息，还是目标可由外部Probe解码、但模型自身决策路径无法利用。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_5_local.py --dry-run
python run_level7_6_6_5_local.py
```

Memory数据集和Probe结果按seed独立保存，可用同一命令续跑；不要加 `--force`。
