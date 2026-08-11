# Level 6.17.1：专用step-2 detector/utility增量实验

本阶段不再把首后继chunk的probe直接重放到第二步。所有轨迹先执行冻结的Level
6.16.1首步策略，然后在第二步采集新的clean/polluted特征以及0.20/0.25动作反事实，
训练专用step-2 detector和utility。

第二步只允许处理首步没有触发的样本，避免重复干预。seed1001的step-2 detector
clean FPR约束收紧到3%，其余模型为5%；最终第一步与第二步的clean联合触发率仍
必须不超过5%。

## 主要终点

- step-2相对冻结dynamic one-step的总体增量95% CI下界大于0；
- seed1001的step-2增量不为负；
- 每模型第一步+第二步clean联合触发率不超过5%。

数据使用全新的2个train、1个validation和1个test随机种子。若独立test没有正
增量，将正式停止多步扩展并保留单步策略。

```powershell
python run_level6_17_1_local.py
```

结果写入`experiments/level6_17_1/formal/`，step-2数据集按模型保存并可续跑。
