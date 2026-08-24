# Level 7.8.5：全Key反事实读出训练

Level 7.8.4.1发现，Memory中存在多个Value信号，但仅改变Query Key时预测在98%–100%的样本中不变；此前随机单Key监督允许模型通过忽略Key获得高于随机的总体准确率。本阶段只修复读出侧，并以Query Key因果切换作为主要终点。

从Level 7.8.4锁定的`stage_load16.pt`出发，冻结写入器、Embedding、前两层及第三层写入参数。只训练第三层`memory_read`、fusion gate、FFN、norm2与输出头。Memory构造在无梯度模式完成，随后复制同一状态进行多次查询。

两条等计算量训练分支为：

- `all_key_counterfactual`：对同一个Memory同时查询全部2个或4个Key；
- `single_key_equal_compute`：每个Memory只选择一个随机Key，并重复成相同查询行数。

两支都按2事实、4事实各训练400步，使用相同seed、Memory样本数和查询行数。完成后复用Level 7.8.4.1的严格评估：每Key每seed 128个样本；配对查询只改变一个Key token；每个切换方向做McNemar检验并在容量内进行Holm校正。

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_1
python run_level7_8_5_local.py --dry-run
python run_level7_8_5_local.py
```

训练每25步保存恢复点，每个评估单元独立保存。中断后重复正式命令即可，不要使用`--force`。最终结果位于`experiments/level7_8_5/formal/result.json`。

只有当全Key分支在严格Key切换标准上通过并优于等计算量单Key分支，才能认为多地址读出形成；单纯总体准确率提升不算成功。
