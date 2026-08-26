# v0.3 Milestone 1: source-span coverage

This diagnostic checks whether a token from the answer span survives source
selection and cross-chunk retention. It performs no training and makes no
language-generalization claim.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_coverage_gate.py --dry-run
python run_v0_3_coverage_gate.py --smoke-test
python run_v0_3_coverage_gate.py --local-files-only
```

The formal protocol evaluates 2, 4, 8, and 16 chunks. Every distance must reach
an 80% span-hit rate. `smoke.json` records the implementation smoke check; the
formal command writes `results.json` alongside it.
