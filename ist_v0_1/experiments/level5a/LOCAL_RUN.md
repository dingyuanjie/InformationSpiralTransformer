# Run the formal Level 5A ablation locally

From the `ist_v0_1` directory:

```powershell
python run_level5a_local.py
```

The default run compares Transformer, IST-A, IST-B and IST-C with seeds 313 and
42. It uses CUDA mixed precision, batch size 16, validation gates, stage
checkpoints and automatic resume. Completed runs are skipped.

If GPU memory is insufficient:

```powershell
python run_level5a_local.py --batch-size 8 --eval-batch-size 8
```

For one model first:

```powershell
python run_level5a_local.py --variants ist-c --seeds 313
```

Results are written to `experiments/level5a/formal/summary.json`. Send that file
back for analysis. Use `--force` only when intentionally replacing existing
runs.
