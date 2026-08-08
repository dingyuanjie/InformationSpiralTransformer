# Level 6.11: selective causal memory intervention

This evaluation-only experiment tests whether Level 6.10's layer localization
and slot redundancy are causal for 16-chunk behavior.

Layer conditions:

- intact memory;
- zero layer 0, 1, or 2 separately;
- preserve only layer 0, 1, or 2.

Final-layer slot conditions preserve 1, 2, 4, 8, 16, or 32 slots after every
chunk. Each count compares the fixed first K slots with three deterministic
random K-slot subsets. All conditions receive the same 400 test examples.

Layer-2 causal localization requires intact query >=90%, zero-layer-2 query
<=20%, and only-layer-2 query >=90%. Slot summaries report the smallest fixed
and random subsets that retain >=90% query.

```powershell
python run_level6_11_local.py
```

Results are stored under `experiments/level6_11/formal/`.
