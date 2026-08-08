# Level 6.9: causal persistent-memory intervention

This is an evaluation-only experiment on the three behaviorally successful
Level 6.8 models (606, 808, and 1001). No optimizer updates are performed.

For identical test examples, four conditions are compared at 2, 4, 8, and 16
chunks:

- `intact`: preserve each sample's memory normally;
- `reset`: pass `None` before every next chunk;
- `zero`: replace all layer memories with zero tensors;
- `batch_roll`: deterministically give every sample another sample's memory.

The batch-roll condition preserves memory tensor distribution and magnitude but
breaks identity, separating information content from generic memory activation.

Each model/chunk condition uses 400 examples. A causal pass requires intact
query >=90%, every intervened query <=20%, and local accuracy >=90% in all
conditions. Probe accuracy is recorded but is not part of the causal gate.

```powershell
python run_level6_9_local.py
```

Results are stored under `experiments/level6_9/formal/`.
