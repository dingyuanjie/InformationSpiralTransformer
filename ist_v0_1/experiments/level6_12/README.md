# Level 6.12: exhaustive final-layer causal slot map

For each successful Level 6.8 model, this evaluation exhaustively tests:

- all 32 keep-one interventions;
- all 32 leave-one-out interventions;
- intact final-layer memory.

Every condition uses the same 400 examples at 16 chunks. The script also
captures final-layer memory-read attention, slot norms, update gates, and the
mean fusion-gate strength. It reports Pearson/Spearman relationships between
read attention and both keep-one sufficiency and leave-one-out necessity.

Per-model and aggregate PNG maps show keep-one query accuracy, leave-one-out
accuracy drop, and read-attention mass for every slot. Progress is saved after
each condition, so long runs resume without repeating finished slot tests.

```powershell
python run_level6_12_local.py
```

Results are stored under `experiments/level6_12/formal/`.
