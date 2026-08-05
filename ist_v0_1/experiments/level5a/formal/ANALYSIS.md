# Level 5A formal gated ablation analysis

## Protocol

All variants used three layers, hidden size 64, RoPE, the same marked-retrieval
data stream, local auxiliary objective, optimizer, batch size 16 and seeds 313
and 42. A stage passed at 90% query accuracy.

## Results

| Variant | Seed | 128 tokens | 256 tokens | 512 tokens | Complete |
|---|---:|---:|---:|---:|---:|
| Transformer | 313 | 90.00% (1200, pass) | 81.88% (500, fail) | - | No |
| Transformer | 42 | 90.62% (1500, pass) | 85.00% (500, fail) | - | No |
| IST-A | 313 | 73.12% (1500, fail) | - | - | No |
| IST-A | 42 | 70.62% (1500, fail) | - | - | No |
| IST-B | 313 | 11.88% (1500, fail) | - | - | No |
| IST-B | 42 | 80.62% (1500, fail) | - | - | No |
| IST-C | 313 | 90.62% (1400, pass) | 100% (500, pass) | 100% (500, pass) | Yes |
| IST-C | 42 | 10.62% (1500, fail) | - | - | No |

## Supported conclusions

- Memory without the explicit Memory-to-Hidden fusion gate (IST-A/B) did not
  improve this task under the tested budget.
- IST-C can solve all three stages and adapts rapidly after the first stage.
- The full IST-C result is highly seed-sensitive: success rate was only 1/2.
- The Transformer was more consistent at 128 tokens, passing for both seeds,
  but 500 steps were insufficient for its 256-token stage.

## Conclusions not supported yet

- These runs do not prove that IST-C outperforms a Transformer. Only one IST-C
  seed completed, and the Transformer was stopped before convergence at 256.
- Mean time and peak memory are not directly comparable because failed runs
  stopped at shorter sequence lengths.
- The diversity loss cannot be called beneficial: IST-B was unstable and never
  passed stage 1 in this experiment.

## Next confirmatory experiment

1. Increase the first-stage ceiling to 3000 steps and later stages to 1000.
2. Require two consecutive validation passes to avoid a noisy one-off gate.
3. Run IST-C and Transformer on five seeds before expanding other variants.
4. Compare fixed-step accuracy and steps-to-90 in addition to gated success.
5. Sweep IST-C diversity weights 0.01, 0.05 and 0.1 if seed instability remains.
