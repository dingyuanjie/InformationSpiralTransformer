# Level 7.6.5：8192 长上下文稳定性与公平效率复验

本阶段不重复训练，冻结读取 Level 7.6.4 的 4096-token 最终 checkpoint，在完全相同的 8192-token 样本上配对比较 `transformer-matched`、`ist-full` 和 `ist-stable`。

正式协议：5 个 seed、9 个距离窗口、每个 seed/窗口 100 个样本；共计每个模型 4500 个样本。输出总体与逐 seed Wilson 95% 区间、逐窗口配对 McNemar 精确检验及 Holm 校正、吞吐和 CUDA 峰值显存。单个窗口完成后立即原子保存，因此中断后直接执行同一命令即可续跑。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_5_local.py --dry-run
python run_level7_6_5_local.py
```

不要使用 `--force`，除非明确需要删除逻辑上的续跑效果并重算所有窗口。正式输出位于 `experiments/level7_6_5/formal/`。
