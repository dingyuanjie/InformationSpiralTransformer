# Level 6.15.2：跨初始化反向机制诊断

## 1. 完成情况

- 模型种子：606、808、1001。
- 每模型 3 个槽位组合 × 2 个污染时点，共 18 个污染设定。
- 每条件 3 个新评估种子 × 400 样本，共 1200 样本。
- 162 个条件全部完成；每个条件同步计算参考干净、matched clean 和污染轨迹。
- 保存了污染后第 0–3 步的逐样本 Memory 动力学以及 corrected/harmed 分组轨迹。

## 2. 主要结果：固定 MemoryAttention 抑制不再产生总体收益

| 干预 | 总体准确率变化 | 正/负设定 | 干净准确率变化 | donor reduction |
|---|---:|---:|---:|---:|
| step 1，scale 0.10 | -0.97 pp | 7 / 11 | -2.81 pp | +1.30 pp |
| step 1，scale 0.15 | -0.25 pp | 9 / 9 | -1.57 pp | +1.02 pp |
| step 1，scale 0.20 | +0.06 pp | 9 / 9 | -1.04 pp | +0.44 pp |
| step 1，scale 0.25 | +0.08 pp | 9 / 9 | -0.63 pp | +0.19 pp |
| step 2，scale 0.20 | -0.45 pp | 7 / 11 | -1.25 pp | +0.72 pp |
| steps 1–2，scale 0.20 | -0.70 pp | 6 / 12 | -2.03 pp | +0.77 pp |

在 Level 6.15.1 中，只干预 MemoryAttention 的 scale=0.25 平均为 +0.48 pp；使用第三批独立评估样本后仅为 +0.08 pp，18 个设定正负各半。固定支路干预没有形成可复现的总体鲁棒收益。

抑制越强，donor attraction 通常下降越多，但准确率和干净性能反而越差。这证明“减少 donor 输出”不等于“修复污染”：干预在移除一部分 donor 错误的同时制造了更多其他类别错误。

## 3. 初始化决定干预方向

### step 1，scale 0.20

| 模型 | 污染准确率变化 | matched-clean 变化 | corrected / harmed 样本总数 | donor reduction |
|---:|---:|---:|---:|---:|
| 606 | +0.10 pp | +0.71 pp | 262 / 255 | -0.03 pp |
| **808** | **-1.83 pp** | **-5.38 pp** | 415 / 547 | +1.66 pp |
| 1001 | **+1.92 pp** | +1.54 pp | 355 / 217 | -0.31 pp |

seed 808 的反向不是偶然的单组合现象：所有测试条件在该模型上平均都为负。干预确实降低了 donor prediction，但 harmed 样本比 corrected 样本多 132 个，并且无污染输入也下降 5.38 pp。因此根因是 **seed 808 对正常 MemoryAttention 写入高度依赖**；固定抑制破坏的有效信息超过它消除的污染信息。

seed 1001 则表现相反：干预增加的 corrected 样本明显多于 harmed 样本，且 matched-clean 不降反升。相同算子在不同初始化中处于不同工作区间。

## 4. 三种初始化具有不同的 Memory 工作区间

污染后的第一个后继 chunk，未干预基线的平均动力学为：

| 模型 | update gate | encoded norm | attended Memory norm | 传播项 / encoded 比率 | 污染相对 L2 |
|---:|---:|---:|---:|---:|---:|
| 606 | 0.176 | 14.13 | 41.49 | 3.83 | 0.349 |
| **808** | 0.172 | **17.71** | **59.54** | 3.77 | 0.448 |
| **1001** | **0.308** | 5.26 | 39.12 | **13.63** | 0.475 |

关键区别不是 seed 808 具有最大的相对传播比例。seed 808 与 606 的传播比率几乎相同（3.77 对 3.83），但 seed 808 的 encoded 和 attended Memory 绝对范数都显著更大；seed 1001 的传播比率则约为它们的 3.6 倍，update gate 约高 1.8 倍。

在 18 个设定上，基线 step-1 指标与 scale=0.20 收益的探索性 Pearson 相关为：

- update gate：`r = +0.72`
- 传播项 / encoded 比率：`r = +0.70`
- encoded norm：`r = -0.79`
- attended Memory norm：`r = -0.71`
- donor projection：`r = +0.10`

因此，干预是否有益主要与模型的正常写入工作区间相关，而不是与 donor 污染强度直接相关。相关性只有 18 个设定且被模型簇结构影响，应视为机制线索而非独立统计证明。

## 5. corrected 与 harmed 轨迹

scale=0.20 后，seed 808 的 harmed 样本在第一个后继 chunk 具有更高 donor projection（约 0.297），corrected 样本约为 0.254；到第三步仍分别约为 0.293 和 0.252。说明 donor 投影较高的样本确实更危险。

但 donor 投影不能单独决定干预是否有益：seed 808 即使 donor reduction 为正，总准确率仍下降。seed 1001 的 corrected 样本 donor projection 持续低于 harmed 样本，但整体净收益为正。最终结果取决于干预对正常递归写入的破坏成本，而不仅是污染幅度。

## 6. 相对范数门控失败

| 相对范数上限 | 总体准确率变化 | seed 606 | seed 808 | seed 1001 | matched-clean 变化 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | -1.57 pp | -0.01 pp | **-6.36 pp** | +1.68 pp | -4.07 pp |
| 0.20 | -1.05 pp | -0.35 pp | **-4.51 pp** | +1.72 pp | -3.21 pp |

该方案没有消除跨初始化反向，反而放大了 seed 808 的损失。原因是未干预传播项/encoded 比率约为 3.8–13.6，而 0.1/0.2 的上限实际构成极强裁剪；相对范数本身也没有区分“有用的大传播”和“有害的大传播”。

因此，“模型内归一化”这一方向不能只做范数裁剪，必须估计内容是否污染以及正常写入的机会成本。

## 7. 更新后的机制结论

1. 污染主要经 MemoryAttention 递归写入扩散这一定位仍成立，但直接削弱该支路不是稳定防御。
2. 不同初始化学习出了功能不同的 Memory 工作区间。seed 1001 是高相对传播、高 update-gate 模式；seed 808 是高绝对激活但低相对传播模式。
3. seed 808 的反向来自正常信息损失：所有强干预都显著破坏 matched-clean，且减少 donor 错误的同时产生更多非 donor 错误。
4. donor projection 可识别更危险的样本，但不能独立决定是否应该抑制写入。
5. 固定 scale、固定余弦阈值和固定相对范数上限三种无训练门控均未跨初始化成功。

## 8. 下一步建议：Level 6.16

停止继续搜索固定门控参数，进入“样本级污染检测与条件干预”：

- 使用冻结模型轨迹训练小型 probe，输入 donor-free 可观测量：update gate、encoded/attended 范数、状态变化、attention entropy、槽位分散度；
- 标签仅由实验阶段的 clean-vs-polluted 轨迹差异生成，推理时不能使用 donor 或干净参照；
- 分模型校准阈值，同时测试 leave-one-model-out，判断检测规则能否迁移到未见初始化；
- 只有 probe 判为高风险时才短时降低 MemoryAttention，并把 matched-clean false positive rate 设为主要约束；
- 主要终点应为净准确率与 harmed-minus-corrected，而不是单独降低 donor attraction。

现阶段不再有证据支持继续手工扫描固定传播比例。下一阶段的关键是先可靠识别“何时这次写入是污染”，再决定是否干预。
