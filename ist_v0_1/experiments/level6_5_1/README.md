# Level 6.5.1: 16-chunk optimization stability

Level 6.5 found that deterministic `hard400` passed 2, 4, and 8 chunks. At 16
chunks it reached 95% query accuracy at step 100 but degraded under continued
training. This experiment isolates the 16-chunk optimizer update.

All variants load the exact same 8-chunk checkpoint. Training uses strict
deterministic algorithms and the mathematical SDP backend; flash and
memory-efficient SDP are disabled because their backward kernels produced a
non-determinism warning during smoke testing.

```text
experiments/level6_5/deterministic/hard400_seed313/stage3.pt
```

They use the same deterministic data stream and differ only in learning rate:

- `1e-4` (control)
- `5e-5`
- `2.5e-5`
- `1e-5`

Each variant trains for 1,000 probe-free 16-chunk steps, followed by another
500 maintenance steps at the same learning rate. The report includes baseline,
peak query, first consecutive 95% gate, final training query, final maintenance
query, and the minimum query over the last five maintenance evaluations.

Run locally:

```powershell
python run_level6_5_1_local.py
```

Results are written to `experiments/level6_5_1/formal/` and completed learning
rates are reused unless `--force` is supplied.
