# Level 6.18：外部初始化验证

Level 6.8原始五种子协议中，seed707在8-chunk门控失败，seed909在16-chunk门控
失败，因此两者没有最终withdrawal检查点。本阶段保留原失败记录，从各自失败阶段
的优化器检查点继续最多3000步，仍要求两次连续query≥95%；通过后执行与Level 6.8
完全相同的withdrawal。

预注册角色：

- seed707：校准模型；
- seed909：完全留出模型。

固定槽位对为`(13,28)`、`(2,7)`、`(10,17)`，不根据新模型结果选择。只在707
上训练单步detector/utility；909比较零样本迁移和只使用clean分数重新校准detector
阈值两种模式。clean重校准不得使用污染标签或动作结果。

外部成功标准为：909 clean重校准模式净收益95% CI下界大于0，且每时点clean
FPR不超过5%。若任一检查点恢复失败，外部验证自动停止并记录负结果。

```powershell
python run_level6_18_local.py
```

结果写入`experiments/level6_18/formal/`。扩展阶段每100步保存，可断点续跑；原
Level 6.8目录不会被覆盖。
