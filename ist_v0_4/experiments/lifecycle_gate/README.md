# v0.4 Milestone 1: memory lifecycle gate

The gate compares incidental exposure, distinctive event encoding, spaced
repetition and retrieval rehearsal across 16/32/64 chunks. The remembered unit
is an exact eight-token source event containing an arbitrary target token.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_lifecycle_gate.py --dry-run
python run_v0_4_lifecycle_gate.py
```

The protocol does not demand that all incidental details survive. It requires
distinctive events to enter episodic memory and rehearsed events to retain their
complete source span and consolidate more reliably than incidental exposure.
Results and tracebacks are saved to `results.json` and `run.log`.
