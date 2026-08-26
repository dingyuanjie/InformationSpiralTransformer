# v0.3 Milestone 2.2.1: strict failure decomposition

This read-only tomography keeps the step-400 checkpoint and strict OOD data
locked. It separates target-token availability in Memory, Reader ranking given
availability, and Decoder behavior when every non-target slot is masked out.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_failure_decomposition.py --dry-run
python run_v0_3_failure_decomposition.py --local-files-only
```

The oracle condition is diagnostic only: it filters an already-built state to
the exact target answer signature and performs no training.
