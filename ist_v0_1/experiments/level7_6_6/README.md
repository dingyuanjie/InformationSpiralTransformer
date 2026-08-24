# Level 7.6.6：16K/32K 冻结外推边界

本阶段不继续训练。将 Level 7.6.4 的4096-stage checkpoint 加载到最大长度32768的同构 RoPE 模型中，在16K和32K上测试冻结零样本外推。

每个长度划分四个相对距离窗口，每个模型、seed、窗口评估50个样本。记录准确率、Wilson 95%区间、吞吐、峰值显存以及OOM边界。某个模型/seed首次OOM后，后续更远条件标记为 `skipped_after_oom`，不会改变 batch、精度或模型以规避失败。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_6_local.py --dry-run
python run_level7_6_6_local.py
```

每个窗口独立保存，可用同一命令续跑；不要加 `--force`。
