# Level 7.8.1：Memory覆盖机制因果确认

Level 7.8发现，第一个Chunk中只写入一次的信息在聚合层面可可靠维持到16 Chunk，到32 Chunk后无法与随机水平区分；但1000 Chunk后临近写入仍达到19.38%，说明主要问题是旧信息覆盖，而不是整个读写器停止工作。

本阶段对覆盖机制进行选择性因果干预。正常更新条件直接使用Level 7.8锁定结果，新增四种条件：

- `freeze_all`：写入第一个Chunk后冻结三层Memory；
- `freeze_l1`：只冻结第一层；
- `freeze_l2`：只冻结第二层；
- `freeze_l3`：只冻结第三层。

冻结条件仍然对每一个干扰Chunk执行完整模型前向，只丢弃指定层产生的新Memory并保留该层先前状态。因此各条件接收完全相同的目标、干扰流与查询，主终点是每个里程碑相对正常更新的逐样本配对准确率变化。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_1_local.py --dry-run
python run_level7_8_1_local.py
```

实验在每个里程碑保存恢复点；中断后重复正式命令即可，不要使用`--force`。最终结果位于`experiments/level7_8_1/formal/result.json`。

解释规则：若`freeze_all`在长寿命里程碑恢复旧目标，覆盖机制得到直接因果确认；若单层冻结可以复现大部分恢复，则该层是主要遗忘位置；若冻结后仍不能恢复，则首个Chunk的Memory本身没有形成足够的目标编码，或者Query无法利用静态状态。
