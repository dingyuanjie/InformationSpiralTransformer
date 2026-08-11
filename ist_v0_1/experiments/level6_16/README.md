# Level 6.16：样本级污染检测与条件干预

本阶段停止搜索统一固定门控，训练冻结模型上的小型线性 probe，只在 probe 判定
当前 Memory 写入高风险时，短时将第三层 MemoryAttention 传播缩放至 0.20。

Probe 只能使用推理时直接可观测的 12 个动力学特征，包括范数、update gate、
Memory 改写幅度和 compression entropy。禁止使用 donor 标签、正确答案或干净轨迹
参照。

协议包含：

- 按评估种子隔离的 train / validation / test；
- validation 上以 clean false-positive rate 不超过 5% 选择阈值；
- 每模型独立校准和 leave-one-model-out 初始化迁移；
- 全新样本上的条件干预；
- seed 808 为预注册困难模型；
- 主要终点为净准确率、corrected/harmed 数量和 matched-clean 误触发率。

```powershell
python run_level6_16_local.py
```

脚本支持检测数据和条件干预两个阶段的断点续跑。正式结果写入
`experiments/level6_16/formal/`。
