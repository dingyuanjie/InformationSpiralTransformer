# Level 5A.2 fixed-compute and parameter-matched comparison

This experiment compares a 64-wide Transformer, an automatically selected
parameter-matched Transformer (currently width 96), and IST-C. Every model uses
the same five seeds, batches and fixed 2000/800/800 optimization steps. There is
no early stopping.

```powershell
python run_level5a2_local.py
```

For a single-run check:

```powershell
python run_level5a2_local.py --variants ist-c --seeds 313
```

Outputs include fixed-step learning curves, Accuracy-AUC, wall time, memory,
throughput and post-training tests at 512, 1024 and 2048 tokens. Completed runs
are skipped unless `--force` is supplied.
