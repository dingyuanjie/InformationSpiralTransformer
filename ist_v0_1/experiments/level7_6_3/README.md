# Level 7.6.3

固定总长度 8192 token，只改变 needle 到最终预测位置的精确距离，绘制 Transformer、Full IST、Stable IST 的 Memory 保持曲线。每距离五个 seed 合计 1000 个配对样本。

```powershell
python run_level7_6_3_local.py --dry-run
python run_level7_6_3_local.py
```

纯评估任务，按模型和 seed 断点续跑。有效距离定义为 Wilson 95% 下界仍高于 1/16 随机准确率的最远注册距离。
