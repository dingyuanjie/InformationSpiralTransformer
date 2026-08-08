# Level 6.5.4: independent-initialization confirmation

This experiment tests the complete formation-to-maintenance pipeline across
five independent model initializations. It no longer holds the formed model
checkpoint fixed.

## Registered protocol

- Model seeds: 313, 42, 2026, 7, 1234
- Random model and probe initialization for every seed
- Stage-1 scaffold: probe-loss weight 0.5 for 400 steps, then zero
- Behavioral gate: query >= 95% in two consecutive evaluations after scaffold
  removal
- Curriculum: 2, 4, 8, and 16 chunks of 128 tokens
- Learning rates: 1e-3 at 2/4 chunks, 2.5e-4 at 8 chunks, and the stabilized
  5e-5 at 16 chunks
- Probe frozen after scaffold removal
- Final 500-step probe-free maintenance at 16 chunks and 5e-5
- Strict mathematical SDP and deterministic CUDA algorithms

The primary outcome is full-pipeline success rate. Stage reach counts separate
formation failures from long-context transfer failures.

Run locally:

```powershell
python run_level6_5_4_local.py
```

Each seed writes its result immediately. Rerunning the command reuses completed
seeds unless `--force` is supplied. Results are stored under
`experiments/level6_5_4/formal/`.
