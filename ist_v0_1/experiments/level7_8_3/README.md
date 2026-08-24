# Level 7.8.3：多次选择性改写耐久性

Level 7.8.2证明，显式标记Oracle门可以先保持目标A，在Chunk 512接纳目标B，再将B保持到Chunk 1000。本阶段把单次切换扩展为十次改写，检验L3是否能够反复接纳新状态，还是在多次写入后逐渐退化。

正式流在Chunk 1、100、200、300、400、500、600、700、800、900写入十个不同目标，目标序列为`target_v=(base+7*v) mod 16`。在每次写入后立即查询，并在下一次写入前查询当前最新目标；最后在Chunk 1000再次查询第十个目标。

三个严格配对条件不变：

- `normal_update`：L3在每个Chunk更新；
- `freeze_l3_after_first`：只保留第一次写入；
- `oracle_marker_l3_gate`：L3只在十个含token 17的写入Chunk更新，L1和L2始终正常更新。

五个seed各32个类别平衡样本。结果记录每次即时获得率、约100 Chunk后的保持率、L3相对第一次及最近一次写入状态的相似度，以及Oracle相对两个对照的逐样本McNemar检验。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_3_local.py --dry-run
python run_level7_8_3_local.py
```

每个里程碑自动保存恢复状态；中断后重复正式命令即可，不要使用`--force`。最终结果位于`experiments/level7_8_3/formal/result.json`。

本实验仍然只要求“最近一次写入”的单状态语义，不代表可以同时寻址十个历史事实。如果十次改写保持稳定，下一阶段才应测试多槽分配与旧版本回查。
