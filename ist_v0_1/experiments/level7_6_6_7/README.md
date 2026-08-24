# Level 7.6.6.7：seed 7 top-4槽位独立确认

本阶段冻结上一阶段由验证集选出的seed 7第三层slots 0、20、24、26。唯一主要检验是删除这四个槽位与intact在合并近/远窗口上的配对差异，因此不需要多条件校正。

近距和最远窗口各使用128个目标类别严格平衡样本。三组预注册、互不重叠的随机四槽删除作为等大小特异性对照；相同索引的第二层删除用于层特异性检查；top-4两倍增强仅作为方向性次要终点。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_7_local.py --dry-run
python run_level7_6_6_7_local.py
```

每个窗口/条件独立保存，可用相同命令续跑；不要加 `--force`。
