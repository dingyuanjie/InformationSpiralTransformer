# Level 7.6

比较参数匹配 Transformer、原始 IST 和冻结 L3 slot queries 的稳定 IST。固定训练预算后，在 1024/2048/4096/8192 token 上测试外推，同时记录显存、吞吐和 OOM。

```powershell
python run_level7_6_local.py --dry-run
python run_level7_6_local.py
```

默认 3 个模型 × 5 个 seed，共 15 个正式训练。每个 stage 保存 checkpoint，可按 stage 断点续跑。
