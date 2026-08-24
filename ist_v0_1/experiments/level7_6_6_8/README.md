# Level 7.6.6.8：跨seed槽位因果复验

本阶段完整复用 Level 7.6.6.7 的独立确认执行器，只将模型替换为seed 2026，并冻结 Level 7.6.6.5 验证集选出的第三层slots 31、3、5、12。

唯一主要检验仍为验证top-4删除与intact在合并近/远窗口上的配对差异。每窗口128个目标类别严格平衡样本。三组不包含任何目标槽位的随机L3四槽删除、L2相同索引删除和L3 top-4两倍增强均为次要特异性对照。

该阶段使用与seed 7确认实验相同的样本生成种子，因此可以比较两个成功初始化的效应方向和幅度；统计检验仍在每个seed内部独立完成。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_8_local.py --dry-run
python run_level7_6_6_8_local.py
```

每个窗口/条件独立保存，可用相同命令续跑；不要加 `--force`。
