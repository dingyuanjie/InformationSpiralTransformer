# Level 7.6.3.1

固定序列长度为 8192，在九个对数距离窗口内随机 needle 位置。每窗口五个 seed 合计 1000 个配对样本；IST 与 Transformer 使用 exact McNemar 检验，并在每个 IST 变体内对九个窗口做 Holm 校正。

```powershell
python run_level7_6_3_1_local.py --dry-run
python run_level7_6_3_1_local.py
```

纯评估任务，按模型和 seed 断点续跑。
