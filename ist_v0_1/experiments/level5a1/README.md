# Level 5A.1 stability confirmation

This experiment compares only the standard Transformer and full IST-C using
five seeds. Stage ceilings are 3000/1000/1000 steps and a stage passes only
after two consecutive validation measurements reach 90% query accuracy.

Run from `ist_v0_1`:

```powershell
python run_level5a1_local.py
```

For an initial single-seed check:

```powershell
python run_level5a1_local.py --variants ist-c --seeds 313
```

Results are saved under `experiments/level5a1/formal/`. Completed runs are
skipped, and stage checkpoints allow interrupted runs to continue.
