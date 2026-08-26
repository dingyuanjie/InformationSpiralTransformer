# v0.3 Milestone 2.2: strict generalization

The step-400 Reader checkpoint is locked. Evaluation replaces training entities,
answer words, fact/query templates and fixed fact position; adds three unrelated
credential facts; and extends evaluation to 32 chunks. No optimizer is created.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_strict_generalization.py --dry-run
python run_v0_3_strict_generalization.py --local-files-only
```

`results.json` is the structured record and `run.log` mirrors terminal output
and tracebacks automatically.
