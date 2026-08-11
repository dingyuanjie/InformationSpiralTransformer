# Level 6.17：早期污染的多步选择性防御

Level 6.16.2 证明冻结策略在 chunk 8 有明确收益，但在 chunk 4 接近中性。本阶段
保持 Level 6.16.1 的 detector、utility probe、阈值和动作规则不变，将同一选择性
决策扩展到污染后的前两个 chunk。

## 完整反事实

每个测试样本同时计算并保存：

- 无防御 baseline；
- 固定 0.20 单步与两步；
- 固定 0.25 单步与两步；
- frozen dynamic 单步；
- frozen dynamic 两步。

污染与 matched-clean 都实际运行所有策略，能够严格配对比较动态动作选择和固定
动作。使用 3 模型 × 3 槽位组合 × 4 新评估种子 × 200 样本，共 7200 个 chunk-4
污染样本。

## 预注册联合标准

- dynamic two-step 相对 baseline 的95% CI下界大于0；
- dynamic two-step 点估计优于 dynamic one-step；
- dynamic two-step 每条件 clean联合触发率不超过5%；
- seed 808 dynamic two-step收益不低于-0.25 pp。

当前 Level 6.8 只有 seed606、808、1001 三个 `withdrawal_phase3.pt`，seed707和909
目录没有对应检查点，因此第四初始化迁移测试不在本级伪造执行。

```powershell
python run_level6_17_local.py
```

结果写入 `experiments/level6_17/formal/`，逐槽位组合保存并支持断点续跑。
