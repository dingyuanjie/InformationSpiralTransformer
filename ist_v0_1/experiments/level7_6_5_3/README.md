# Level 7.6.5.3：随机交错顺序与长时稳态复验

本阶段固定 seed 313、8192 token、batch=1 和相同的 20 个输入，同时加载三个模型。在各自预热 10 次后执行 12 轮测试：六种模型顺序各重复两次后用固定随机种子打乱，使每个模型在每个位置恰好出现四次，从而消除测试位置、GPU 升频和热状态对单个模型的系统性偏向。

每个 block 连续执行 20 次前向，同时记录 Python 墙钟、CUDA Event，以及计时区间外的 GPU 温度、功耗、SM/显存时钟和利用率。结果给出 block 延迟中位数、IQR、CV、不同测试位置的中位速度，以及逐轮配对的 IST 加速比。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_6_5_3_local.py --dry-run
python run_level7_6_5_3_local.py
```

每个 block 独立保存；中断后执行相同命令续跑，不要加 `--force`。
