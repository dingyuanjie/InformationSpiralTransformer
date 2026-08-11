# Level 6.16.2：冻结协议的扩大样本独立确认

本阶段冻结 Level 6.16.1 的 detector、utility probe、模型×时点阈值与动作选择
规则。脚本不训练 probe、不重新校准阈值、不搜索参数，并将冻结输入文件的 SHA-256
写入预注册文件。

每个模型×时点使用 3 个预注册槽位组合 × 4 个新评估种子 × 200 样本，共
2400 个全新样本。污染轨迹和 matched-clean 轨迹都实际执行冻结策略，同时评估
动作 0.20/0.25 的反事实。

## 联合成功标准

- 总体 accuracy gain 的 bootstrap 95% CI 下界大于 0；
- seed 808 合并点估计不低于 -0.25 pp；
- 每个模型×时点×槽位组合的 clean 触发率不超过 5%；
- 总体 corrected 多于 harmed，且配对 McNemar `p < 0.05`。

```powershell
python run_level6_16_2_local.py
```

正式结果写入 `experiments/level6_16_2/formal/`，脚本逐条件保存并支持断点续跑。
