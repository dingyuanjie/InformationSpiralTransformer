# v0.3 Milestone 2.1: Reader alignment

The Qwen backbone and Writer salience policy remain frozen. Training updates
only Query/Key/Value/Output, QueryNorm, and the residual injection gate with
answer, retrieval, and alternating zero/swap causal losses.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_reader_alignment.py --dry-run
python run_v0_3_reader_alignment.py --local-files-only
```

Resume an interrupted run with `--resume`. Checkpoints are unique, immutable
`.pt` files (ignored by Git), avoiding Windows atomic-replace permission errors.
All output and tracebacks are mirrored to `run.log`.

After completion, evaluate the trained Reader without changing the locked gate:

```powershell
python run_v0_3_retrieval_gate.py --local-files-only --checkpoint experiments\reader_alignment\reader_step_000400.pt
```
