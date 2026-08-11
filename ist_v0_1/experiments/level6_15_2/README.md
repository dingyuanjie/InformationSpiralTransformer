# Level 6.15.2：跨初始化反向机制诊断

本阶段把 seed 808 预注册为困难模型，诊断固定传播抑制为何在该初始化上反向。

## 记录的污染后轨迹

- 第三层 Memory 相对 L2、与干净轨迹的余弦；
- donor 方向投影；
- update gate；
- 每步 Memory 改写幅度与新旧状态余弦；
- encoded token 与 attended Memory 范数；
- 实际传播项相对 encoded token 的范数；
- 由错误变正确和由正确变错误样本的分组轨迹。

## 干预

- 只干预已定位的 MemoryAttention 写入支路；
- 首个后继 chunk 固定比例 0.10、0.15、0.20、0.25；
- scale=0.20 的第二步单独干预与连续两步干预；
- 模型内相对范数上限 0.10、0.20，检验跨初始化归一化方案。

所有条件使用新的评估种子并同时计算 matched clean 轨迹。脚本支持断点续跑。

```powershell
python run_level6_15_2_local.py
```

正式结果写入 `experiments/level6_15_2/formal/`。
