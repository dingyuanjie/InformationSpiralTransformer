# Level 5A.2 fixed-compute, parameter-matched analysis

All models used five identical seeds and fixed 2000/800/800 optimization steps
at lengths 128/256/512. No early stopping was used.

| Variant | Parameters | Accuracy-AUC | 512 | 1024 | 2048 | Time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer-64 | 152,403 | 0.578 ± 0.114 | 95.5 ± 5.4% | 50.0 ± 14.0% | 13.5 ± 11.0% | 56.7 s | 93.9 MB |
| Transformer-96 | 339,187 | 0.838 ± 0.039 | 100 ± 0% | 44.0 ± 15.5% | 12.0 ± 7.4% | 78.9 s | 1205.5 MB |
| IST-C | 345,171 | 0.616 ± 0.107 | 92.0 ± 13.0% | 53.5 ± 20.1% | 17.5 ± 7.9% | 123.7 s | 165.5 MB |

## Interpretation

- At matched parameter count and fixed training steps, the wider Transformer is
  decisively better inside the trained context range: higher AUC, perfect
  five-seed 512-token accuracy, lower variance and lower wall time.
- IST-C does not currently justify its added complexity on the trained synthetic
  retrieval task. It is slower than both baselines and less stable than the
  parameter-matched Transformer.
- IST-C has higher mean zero-shot accuracy beyond the trained length (53.5% vs
  44.0% at 1024; 17.5% vs 12.0% at 2048), but n=5 uncertainty is large and this
  is not yet strong evidence of superior extrapolation.
- Transformer-96's 1.2 GB peak is likely an SDPA kernel/head-dimension artifact:
  head dimension 12 can select a memory-heavy attention path at length 2048.
  It should not be interpreted as a fundamental Transformer memory cost without
  a kernel-controlled benchmark.

## Decision

Do not claim that IST-C beats Transformer. The supported claim is narrower:
IST-C learns the task and may have a weak out-of-range transfer signal, while a
parameter-matched Transformer is substantially more sample-efficient and stable
within the trained range.

## Next experiment

Stop increasing full-window length. Test the actual architectural hypothesis:
fixed 512-token chunks with memory carried across chunks. Compare Transformer,
IST-C, and IST-C with memory reset. This isolates persistent Spiral Memory from
ordinary full-context attention and avoids the current parameter-efficiency
disadvantage being mistaken for long-term memory capability.
