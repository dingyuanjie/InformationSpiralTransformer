# Level 5A.1 five-seed stability analysis

Both Transformer and IST-C completed all three gated stages for all five seeds.
The earlier seed failures were therefore primarily insufficient-budget failures,
not permanent optimization failures.

| Variant | Success | Mean steps L128 | L256 | L512 | Mean time | Peak memory | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 5/5 | 1600 | 760 | 640 | 62.7 s | 93.9 MB | 152,403 |
| IST-C | 5/5 | 2000 | 640 | 620 | 121.0 s | 165.5 MB | 345,171 |

Mean validation accuracy at the second consecutive passing measurement was
91.88/94.00/97.88% for Transformer and 94.75/97.88/98.50% for IST-C at lengths
128/256/512. These are gate-exit measurements, not fixed-compute comparisons.

## Interpretation

- Transformer established the initial retrieval circuit 20% sooner (1600 vs
  2000 steps) and used much less compute.
- After the initial circuit existed, IST-C adapted to 256 and 512 tokens in
  slightly fewer steps (640/620 vs 760/640). This is evidence of a possible
  curriculum-transfer benefit, but the 512 difference is small.
- IST-C required 2.26x parameters, 1.76x peak memory and 1.93x wall time.
- The present results do not establish overall superiority for IST-C. They show
  reliable learning and a candidate distance-transfer effect that requires a
  fixed-compute and parameter-matched confirmation.

## Required next experiment

Evaluate both models at identical checkpoints and total optimization steps,
without early-stop differences. Add a parameter-matched wider Transformer.
Report accuracy at fixed steps, accuracy-AUC, throughput and 512-to-2048
transfer. Only then can the memory module's benefit be separated from extra
parameters and compute.
