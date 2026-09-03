# Information Spiral Transformer（IST）

IST 是一个探索“有限状态、跨 Chunk、可干预持久记忆”的实验项目。它不使用不断扩大的注意力窗口保存全部历史，而是研究模型能否在流式输入中完成选择性写入、长期保留、按查询读取，以及有限容量下的主动遗忘。

当前结论不是“IST 已全面超过标准 Transformer”。更准确地说：项目已经在合成跨段任务上证明了持久 Memory 的存在及因果可用性，并定位了多个真实瓶颈；但在冻结预训练语言模型上的开放词汇、实体绑定和严格分布外泛化仍未解决。

## 架构演进

| 版本 | 核心设计 | 得到的主要认识 | 状态 |
|---|---|---|---|
| `ist_v0_1` | 多层递归 Memory，与小型 Transformer 端到端训练 | 合成跨 Chunk 任务中形成了分布式、持久且可因果干预的 Memory；主要瓶颈逐步转向读取和输出映射 | 机制证据成立，通用性未证明 |
| `ist_v0_2` | 分层快/慢 Memory，并接入冻结 Qwen 0.5B | 简单外接 Memory 可以保存信号，但不能可靠转换为自然语言能力；多轮冻结干预没有稳定优于 baseline | 路线关闭，保留负结果 |
| `ist_v0_3` | 保存真实 token 状态与来源位置，稀疏 Query top-k Reader | Writer 覆盖可以通过，但 Reader、干扰抑制、开放答案解码和容量会分别失败 | 严格 OOD gate 失败 |
| `ist_v0_4` | 工作/情景/语义三层生命周期，事件级写入、强化与遗忘 | 已能控制“记什么、忘什么”，并解决关系被切断的问题；实体条件 Query 检索尚未学成 | 当前开发版本 |
| `ist_v0_5` | 可溯源多向量 Evidence＋递归 Core 双通路 | 已完成严格新绑定协议、因果干预与 Level A 执行器；尚无正式实验结果 | 当前候选架构 |

## 已被实验支持的结论

### 1. v0.1：Memory 不是完全无效的旁路

在测试过的合成跨 Chunk 任务中，Memory 中存在持久、分布式且可因果使用的信息。代表性锚点：

- 污染防御干预带来总体 `+0.52` 个百分点收益，bootstrap 95% CI 为 `[+0.23, +0.80]`。
- 仅替换输出头时，12-chunk 准确率从 `89.99%` 提升至 `96.19%`，16-chunk 从 `83.25%` 提升至 `91.31%`，说明部分错误发生在 Memory 之后。
- 在 277 个源错误样本中，最终层 Memory probe 仍有 `81.95%` 准确率，说明信息经常“已经存在，但部署的读取路径没有取出来”。
- Probe 剂量配合 Oracle 方向可恢复 `94.38%`，冻结的无标签 router 只恢复 `21.35%`；后续因子化修复为 `24.41%`，未达到预注册的 `25%` 门槛。因此 router 修复分支按协议关闭。

完整证据、置信区间和边界见 [`ist_v0_1/experiments/level7_0/EVIDENCE_LEDGER.md`](ist_v0_1/experiments/level7_0/EVIDENCE_LEDGER.md)。这些结果支持特定合成任务中的机制结论，不支持普遍优于标准 Transformer 的主张。

### 2. v0.2：保存一个向量不等于获得语言记忆

v0.2 将分层 Memory 接入冻结的 Qwen 0.5B，依次测试单事实、实体绑定、状态更新、冲突、核心细节、多跳以及不同层注入。早期同分布测试会出现高分，但锁定模板、答案、实体或干扰项后不稳定。最终冻结 Memory 干预中，最优条件仍是 `baseline`：

- `baseline`：增益 `+0.0000`
- `prototype_pc1_topk4`：增益 `-0.0221`
- `prototype_center`：增益 `-0.0286`

因此 v0.2 的负结论是：粗粒度压缩、外接 adapter 和局部训练不足以让冻结语言模型形成可泛化的实体—答案绑定。

### 3. v0.3：Writer、Reader、Decoder 是三个独立关卡

v0.3 改为保留真实 token 隐状态及来源信息。Writer coverage gate 在 2/4/8/16 chunks 的 span-hit 分别为 `100% / 100% / 100% / 93.75%`，说明目标 token 通常确实写进了 Memory。

但未训练 Reader 的检索 gate 失败；训练后的 Reader 在训练分布可达 `100%`，严格 OOD 测试却失败：

| 距离 | Reader fact hit | 答案准确率 |
|---:|---:|---:|
| 4 chunks | 50% | 20% |
| 8 chunks | 60% | 10% |
| 16 chunks | 50% | 30% |
| 32 chunks | 20% | 0% |

故障分解显示：4/8/16 chunks 的 Writer availability 均为 `100%`，32 chunks 降为 `30%`。因素消融表明：

- 新 query 模板不是主要问题，单独更换模板仍有 `90%` 准确率。
- 更换未见答案词后降至 `20%`，Oracle 也只有 `20%`，暴露输出解码/开放词汇问题。
- 加入干扰事实后准确率降至 `20%`，但 Oracle 为 `100%`，暴露 Reader 的干扰抑制问题。
- 32 chunks 把容量从 64 增至 128/256 能恢复 Writer availability 到 `100%`，却未恢复答案准确率，说明容量不是唯一瓶颈。

### 4. v0.4：生命周期有效，但检索表示仍失败

v0.4 不再把所有片段同等保存，而是引入工作记忆、情景记忆和语义原型：新近事件先进入工作记忆；新颖或意外事件进入情景记忆；被反复访问的事件增强并可巩固；低价值、老旧、冗余事件竞争淘汰。

合成 lifecycle gate 在 16/32/64 chunks 全部通过：偶发低价值事件最终被遗忘；重复或检索强化事件被保留；强化事件的语义巩固率为 `100%`。这证明生命周期规则按设计工作，但不是自然语言能力证明。

冻结 Qwen Writer 的首轮 gate 在 32 chunks 失败。配对 tomography 找到两个关键现象：

- 固定强化、64 个 episodic slots 时，4/16/32 chunks 的情景保留和语义精确引用均为 `100%`。
- 相对强化也能保留目标，但 utility 从约 `413` 膨胀到 `45,791`，存在明显数值失控，因此没有选作默认机制。

最初的 8-token event 会切断实体—答案关系。关系覆盖实验得到：

| Event 划分 | 完整关系覆盖率 |
|---|---:|
| span 8 / stride 8 | 6.25% |
| span 16 / stride 16 | 46.88% |
| span 16 / stride 8 | 92.19% |
| span 24 / stride 8 | 100% |

因此当前默认使用重叠的 `24/8` event。它修复了“关系有没有完整写入”，但最新 Query/Key 对齐仍没有修复“能不能按实体找回来”：

| 验证距离 | Writer 完整关系可用率 | Query Top-1 |
|---:|---:|---:|
| 4 chunks | 100% | 25.00% |
| 8 chunks | 93.75% | 28.13% |
| 16 chunks | 81.25% | 25.00% |

这是四选一任务，随机线为 `25%`。最后 100 个训练 step 的平均准确率为 `19%`，平均关系可用率为 `94.5%`，且无 NaN。由此可把当前主要故障定位到 Query/Event 表示与匹配，而不是单纯的 Writer 缺失。

## 与标准 Transformer 的关系

仓库包含同参数或受控条件下的标准 Transformer 对照、长上下文速度/显存测试及大量消融。IST 在若干合成长距离任务和流式推理条件下表现出更低的状态增长成本，并在部分任务上取得更高准确率；但这些结果不能外推成“IST 普遍超过 Transformer”。目前尚缺少：

- 多个标准自然语言数据集上的统一训练与盲测；
- 等参数、等训练 token、等 FLOPs 的完整比较；
- 多初始化统计和置信区间覆盖所有关键结论；
- 真实开放词汇生成质量与超长流式部署验证。

所以当前最可靠的项目定位是：**一个具有明确机制证据和完整负结果记录的持久 Memory 研究原型，而不是成熟的 Transformer 替代品。**

## 当前瓶颈与下一步

当前单向量检索把一个 24-token event 直接平均，并主要使用 query 最后一个 token 形成查询。这可能稀释实体身份，也难以表达实体—属性—值的绑定。下一阶段应先做表示 tomography，而不是继续盲目增加训练步数：

1. 比较 query 最后 token、实体 token pooling 和 learned attention pooling。
2. 比较 event mean pooling、token max、attention pooling。
3. 加入词法实体 Oracle，测量 Writer 完整时的检索上限。
4. 若 Oracle 成功，改为多向量 late interaction，让 query 实体 token 与 event 内 token 直接匹配。
5. 冻结 Writer、生命周期和输出侧，仅改变检索表示，以保持因果归因清晰。

## 快速复现

环境与各里程碑参数以对应目录的 README 为准。当前 v0.4 主线可按下列顺序检查：

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_smoke.py
python -m pytest tests -q
python run_v0_4_lifecycle_gate.py
python run_v0_4_pretrained_writer_gate.py --local-files-only
python run_v0_4_paired_tomography.py --local-files-only
python run_v0_4_relation_coverage_gate.py --local-files-only
python run_v0_4_relational_query_alignment.py --local-files-only
```

主要入口：

- [`ist_v0_1` 实验说明](ist_v0_1/experiments/README.md)
- [`ist_v0_2/README.md`](ist_v0_2/README.md)
- [`ist_v0_3/README.md`](ist_v0_3/README.md)
- [`ist_v0_4/README.md`](ist_v0_4/README.md)
- [`ist_v0_5/README.md`](ist_v0_5/README.md)
- [`时间递归链分析.md`](时间递归链分析.md)

## 结果解释原则

- Smoke test 只验证代码路径，不作为科学结论。
- “信息存在于 Memory”不等于 Reader 能找到，也不等于 Decoder 能生成答案。
- Oracle 只用于定位上限，不计作部署性能。
- 失败、近阈值失败和成功结果都保留；不把失败四舍五入为成功。
- 在完成真实数据、等计算量、多 seed 比较前，不声称普遍优于标准 Transformer。
