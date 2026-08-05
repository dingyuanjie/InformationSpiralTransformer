# Level 6.1 minimal two-chunk diagnostic

This diagnostic uses two 128-token chunks, per-layer persistent memories and a
shared linear probe that must decode the target from memory after both chunks.

```powershell
python run_level6_1_local.py
```

Pass criteria require query, memory-probe-after-chunk-1 and
memory-probe-after-chunk-2 accuracy all to exceed 95% twice consecutively.
