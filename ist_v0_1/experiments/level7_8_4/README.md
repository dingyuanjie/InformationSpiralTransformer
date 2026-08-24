# Level 7.8.4：多地址事实容量

Level 7.8.3证明Oracle L3写门可以在1000 Chunk中完成十次“最新值”改写而不累计退化。本阶段不再只查询最新值，而是要求模型同时保存多个Key–Value事实，并在16个干扰Chunk后按Key回查。

写入格式为`[token17, Key, Value]`，查询格式为`[token18, Key, token16]`。Value为0–15类别；同一样本内不同Key使用不重复的随机Value。容量课程依次为1、2、4、8、16个事实，每档训练200步。使用此前已预注册为成功持久Memory组的Seed 2026和Seed 7，从各自Level 7.6.4 checkpoint独立微调。

训练使用Oracle标记L3写门：L3只在写入Chunk更新，16个干扰Chunk保持。完成课程后，同一checkpoint在三种推理路由下评估：

- `oracle_marker_l3_gate`：只在标记写入时更新；
- `normal_update`：每个Chunk都更新；
- `freeze_l3_after_first`：只允许第一个事实进入L3。

每个seed、容量和条件评估64个样本。主要终点是Oracle条件下合并两个seed后，Wilson 95%下界仍高于`1/16`随机水平的最大事实数；另外用逐样本McNemar检验比较两种失败对照。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_4_local.py --dry-run
python run_level7_8_4_local.py
```

训练每25步保存恢复点，每个评估单元独立保存。中断后重复正式命令即可，不要使用`--force`。正式结果位于`experiments/level7_8_4/formal/result.json`。

这是面向Key–Value任务的微调容量测试，不是原checkpoint的零样本能力。成功意味着固定L3状态能够学习多地址存储；失败仍需区分写入格式未学会、槽位分配失败和查询寻址失败。
