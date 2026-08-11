# Level 6.15.1：关键窗口与剂量精细复验

本阶段使用不同于 Level 6.15 的评估种子，独立复验污染后首个后继
chunk 的短时传播抑制，并进一步定位剂量、时间窗口和传播支路。

## 预注册主要假设

污染后的第一个 chunk 同时将 `SpiralAttention` 历史读取和
`MemoryAttention` 历史传播缩放至 0.25，应在三个独立模型种子上均提高准确率，
且 matched-time 干净干预不应产生明显性能损失。

## 实验族

- 剂量：0.10、0.20、0.25、0.30、0.40；
- 时间：污染后第 1、2、3 个 chunk；
- 持续时间：单步与连续两步；
- 支路拆分：仅抑制 SpiralAttention、仅抑制 MemoryAttention、同时抑制；
- 每种干预都有无污染 matched-time 对照。

每个条件包含 3 个新评估种子，每个种子 400 个样本。脚本逐条件保存，支持
中断后直接重新运行续跑。

```powershell
python run_level6_15_1_local.py
```

正式结果写入 `experiments/level6_15_1/formal/`。
