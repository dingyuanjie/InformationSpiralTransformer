# Level 7.8：Memory容量—寿命曲线（单信息寿命）

本阶段回答一个比32K吞吐更基础的问题：IST在固定32槽Memory中，能把一次写入的信息保留多少个Chunk。

正式协议使用Level 7.6.4的五个`ist-full` checkpoint，Chunk长度512，每个seed使用32个类别平衡样本。目标只在第一个Chunk写入一次，随后输入不含标记的随机干扰流，并在1、2、4、8、16、32、64、128、256、512、1000个Chunk处查询。

每个里程碑有三种配对条件：

- `early`：查询第一个Chunk写入的旧目标；
- `late`：在当前状态后立即写入一个不同目标再查询，用来确认读取器此时仍能工作；
- `reset`：不传入Memory直接查询，用来测量类别先验/随机基线。

程序只运行一次1000-Chunk主轨迹，在各里程碑复制Memory做旁路查询；旁路不会污染后续主轨迹。每个里程碑保存可恢复状态，同时记录三层Memory相对首个Chunk状态的余弦相似度、槽位冗余度和范数。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_local.py --dry-run
python run_level7_8_local.py
```

中断后重复正式命令即可从最近的里程碑继续。不要使用`--force`。最终结果位于`experiments/level7_8/formal/result.json`。

Level 7.8只测单条信息寿命。若旧信息能稳定越过较长区间，再进入Level 7.8.1测试1/4/8/16/32条独立事实的容量—寿命二维曲线，避免把“单条记忆会衰减”和“多事实发生容量竞争”混为一谈。
