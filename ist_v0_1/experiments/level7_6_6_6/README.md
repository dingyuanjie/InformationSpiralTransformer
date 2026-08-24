# Level 7.6.6.6：Probe引导的槽位因果路由

本阶段严格使用 Level 7.6.6.5 验证集选择槽位，不读取测试准确率进行选择。每个seed冻结一个最佳槽位和验证排名前4槽位，然后在32K近距与最远窗口执行定向保留、删除和2倍增强。

主要恢复对象是seed 1234；seed 313作为“内容缺失”负对照，seed 2026、7作为高表现因果对照，seed 42作为中间对照。记录准确率、目标概率、logit margin及Probe所选top-4槽位的实际读取质量，并进行逐seed配对McNemar检验和六条件Holm校正。

`SpiralBlock`新增的显式槽位保留、删除和缩放控制默认均关闭，不改变正常推理。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_6_local.py --dry-run
python run_level7_6_6_6_local.py
```

每个seed、窗口、条件独立保存，可用相同命令续跑；不要加 `--force`。
