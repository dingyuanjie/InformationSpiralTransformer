# Level 7.7.1：拓扑感知的自适应 Bank Dropout

Level 7.7表明固定`k=8, p=0.5`会恢复部分读出断联初始化，但也会破坏已经形成的有效Memory结构。本阶段保持相同计算量、训练样本和32K评估样本，仅在当前Memory槽位冗余超过预注册的逐层门槛时允许Bank Dropout触发。

三层门槛为`[0.189, 0.165, 0.169]`，来自Level 7.6.6.1中高表现组与低表现组的近/远窗口`slot_abs_cosine_offdiag`均值中点，未使用Level 7.7结果拟合。超过门槛后仍以50%概率遮蔽8/32个槽位；评估模式完全关闭。

本执行器只训练新的自适应分支，并读取Level 7.7已经锁定的对照组和固定Dropout结果做逐样本配对比较，避免无意义地重复训练。正式协议仍为五个seed、每支续训200步、32K近距/远距各100个平衡样本。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_7_1_local.py --dry-run
python run_level7_7_1_local.py
```

训练每25步保存恢复点，可以直接重复同一条正式命令续跑。不要添加`--force`，除非明确需要清除逻辑上的断点复用并完整重跑。

最终输出位于`experiments/level7_7_1/formal/result.json`，并额外记录每个seed、每一层的平均冗余度、符合门槛比例及实际Dropout触发比例。
