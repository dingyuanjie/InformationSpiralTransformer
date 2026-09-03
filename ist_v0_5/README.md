# Information Spiral Transformer v0.5

v0.5 is a new, isolated research branch for a **Hybrid Evidence–Core Recursive Memory**. It keeps exact, source-traceable evidence spans and a separate fixed-size recursive Core state. The immediate objective is strict held-out binding generalization, not training-set accuracy.

## What is implemented

- fixed-capacity multi-vector Evidence Memory;
- token IDs, absolute positions, source chunks, age, use count and importance metadata;
- MaxSim content-addressed Evidence Reader;
- independently gated recursive Core State;
- hard writer competition with novelty, redundancy, age and usage terms;
- shared-vocabulary/new-binding train-test splits;
- automatic leakage audit;
- no-Memory, last-k, Core-only, Evidence-only and hybrid variants in one parameter envelope;
- zero/reset, Evidence/Core deletion, swap, shuffle, source deletion, Writer/Reader block and identity corruption;
- JSON, CSV, Markdown and optional PNG reporting;
- latency and CUDA peak-memory measurement;
- CPU smoke tests and unit tests.

The implementation is Level A only. The first formal Level A run completed but did not pass the multi-seed stability gate. Qwen 0.5B bridging and semi-natural language Level B remain blocked.

## Audit and design

- [`AUDIT.md`](AUDIT.md) records the v0.1–v0.4 evidence state, v0.4 data flow and leakage/fairness risks.
- [`DESIGN.md`](DESIGN.md) defines the chosen design and causal interventions.
- [`configs/level_a.json`](configs/level_a.json) is the formal multi-seed protocol.

## Run

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_5
python check_splits.py
python -m pytest tests -q
python run_level_a.py --dry-run
python run_level_a.py --smoke-test
```

Only after the smoke path passes, start the formal Level A comparison:

```powershell
python run_level_a.py
```

Results are written under `results/v0_5/level_a`. Checkpoints are excluded by the repository `.gitignore`; compact JSON/CSV/Markdown metrics remain suitable for GitHub.

Run an untrained scaling benchmark independently with:

```powershell
python benchmark.py --chunks 2 4 8 16 32
```

## Success gate

Chance is 6.25% because every entity and every value token occurs during training, while the tested entity–value pair does not. A stage is not successful unless multiple seeds beat the strongest fair baseline on strict held-out bindings and Memory destruction causes the expected drop. Smoke results never satisfy this gate.

## First formal result

All 15 runs completed. Hybrid achieved mean strict held-out accuracy of `39.06% / 21.09% / 12.50%` at 2/8/32 chunks versus a 6.25% uniform-value guess, but seed variance was very high. Evidence destruction eliminated the usable long-range signal in the successful seeds, while Core destruction did not. The honest verdict is **partial Evidence signal, failed stability gate**. See [`results/v0_5/level_a/ANALYSIS.md`](results/v0_5/level_a/ANALYSIS.md).

## v0.5.1

v0.5.1 separates capacity and Reader failures. It forces the exact supervised source occurrence into Oracle Evidence, reports accuracy conditional on exact retention, scans capacities `4/8/12/16/24/32/64`, compares current and stabilized Reader initialization over five seeds, and replaces span reversal with entity/value rebinding.

```powershell
python run_v0_5_1.py --dry-run
python run_v0_5_1.py --smoke-test
python run_v0_5_1.py
```

Formal outputs are written to `results/v0_5_1`. Do not interpret the Oracle score as deployable performance; it is a Writer-bypassed Reader diagnostic.
