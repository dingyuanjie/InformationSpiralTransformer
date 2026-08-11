# Level 6.16.1：时间校准与风险—收益决策

本阶段将 Level 6.16 的污染检测器按模型和 chunk 时点分别校准，并增加 utility
probe，预测一次 MemoryAttention 干预更可能修好还是弄坏当前样本。

## 决策规则

只有同时满足以下条件才干预：

1. 时间校准 detector 判定污染风险超过阈值；
2. utility probe 预测干预收益超过验证集选择的阈值。

系统在 `scale=0.20` 与 `scale=0.25` 中选择预测收益较高的动作。所有 probe 都只
使用推理时可观测动力学，不使用 donor、答案或 clean reference。

## 预注册联合成功标准

- seed 808 独立测试净收益不小于 0；
- chunk 4 和 chunk 8 各自的 matched-clean 触发率都不超过 5%；
- 三模型总体独立测试净收益大于 0。

Utility train / validation / test 使用与 Level 6.16 不重合的新随机种子，脚本支持
utility 数据采集阶段断点续跑。

```powershell
python run_level6_16_1_local.py
```

正式结果写入 `experiments/level6_16_1/formal/`。
