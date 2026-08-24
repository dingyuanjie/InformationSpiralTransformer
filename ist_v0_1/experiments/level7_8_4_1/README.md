# Level 7.8.4.1：逐Key容量确认

Level 7.8.4按合并准确率确认4事实高于随机，但逐Key诊断显示2事实分配均匀，4事实中的Key 2较弱。本阶段锁定已训练的`stage_load16.pt`，不继续训练、不调整超参数，扩大逐Key样本并加入Query Key切换对照。

只复验2事实与4事实。Seed 2026和Seed 7的每个Key各评估128个新样本，合并后每Key共256个样本。严格容量标准要求每一个Key的Wilson 95%下界都高于`1/16`随机水平。

对每一个已经构造好的Memory，程序执行两次仅相差一个Key token的查询：先查询Key k，再把Query改成`(k+1) mod load`。切换成功要求输出更符合新Key对应的Value，而不是继续泄漏旧Key的Value。严格成功要求每个切换方向都高于随机，并通过逐样本McNemar检验；同一容量内按Holm方法校正多个切换方向。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_4_1_local.py --dry-run
python run_level7_8_4_1_local.py
```

每个seed/load/key评估单元独立保存，中断后重复正式命令即可。不要使用`--force`。最终结果位于`experiments/level7_8_4_1/formal/result.json`。
