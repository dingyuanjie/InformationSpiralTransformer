# Level 7.7：机制驱动的训练稳定化

本阶段把Level 7.6.6发现的冗余Memory bank机制转化为训练干预。从每个Level 7.6.4最终checkpoint分叉两条等计算量路线：正常续训，以及结构化bank dropout续训。

Bank dropout在训练模式下对每层以50%概率随机遮蔽8/32个Memory槽位的读取；评估模式自动关闭。每一步在生成训练样本前使用固定seed重置随机数，使两个分支看到完全相同的4096-token训练样本。两支均续训200步，然后在32K近距和最远窗口各评估100个类别平衡样本。

主要终点是五seed配对准确率变化和Wilson下界高于随机水平的成功seed数量。该阶段检验结构化槽位失活能否主动促进类似seed 2026的冗余编码并降低初始化失败率。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_7_local.py --dry-run
python run_level7_7_local.py
```

训练每25步保存恢复点；每个分支、seed和窗口均可独立续跑。不要加 `--force`。
