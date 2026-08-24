# Level 7.8.2：选择性L3写入门控

Level 7.8.1证明，冻结L3可以让首个Chunk形成的信息无衰减地保持到1000 Chunk；但永久冻结不能接收后续重要信息。本阶段使用显式写入标记构造Oracle门，检验“只在重要Chunk更新L3”是否能同时实现长期保持和中途改写。

正式流包含两次写入：Chunk 1写入目标A，Chunk 512写入不同的目标`B=(A+7) mod 16`。查询在改写前要求A，从Chunk 512开始要求B。三个严格配对条件为：

- `normal_update`：L3在每个Chunk更新；
- `freeze_l3_after_first`：L3在第一次写入后永久冻结，预期保住A但不能接纳B；
- `oracle_marker_l3_gate`：L3仅在当前Chunk包含写入标记token 17时更新，普通干扰Chunk保持；L1和L2始终正常更新。

五个seed各使用32个类别平衡样本。主要里程碑为1、16、128、511、512、513、640、1000 Chunk，并记录当前期望目标、旧目标A、新目标B以及L3相对两次写入状态的漂移。Oracle门与两个对照逐样本配对。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_2_local.py --dry-run
python run_level7_8_2_local.py
```

每个里程碑自动保存恢复点。中断后重复正式命令即可，不要使用`--force`。最终结果位于`experiments/level7_8_2/formal/result.json`。

成功判据不是100%准确率，而是同时满足：改写前A长期高于随机水平；Chunk 512后B高于随机水平并优于永久冻结；到Chunk 1000仍优于正常更新。该阶段使用任务标记作为Oracle，不声称已经学会自主判断重要性。
