# Level 6.18.3 formal analysis

## Decision

**Level 6.18.3 passes every preregistered gate.** A change restricted to the
first 16 rows of `model.output` recovers the failed seed-707 checkpoint at 12
chunks, preserves and improves its 8-chunk behavior, and retains sample-specific
Memory causality. This is causal confirmation of the Level 6.18.2 output-head
alignment diagnosis for this seed/checkpoint.

It does not retroactively change the formal Level 6.18.1 failure and does not
yet establish cross-initialization recovery.

## Head fitting and validation selection

The head was trained only on frozen 12-chunk query-token hidden states. Its
12-chunk validation accuracy increased from 91.02% at initialization to 96.68%
at epoch 27; early stopping completed after epoch 33.

The fixed interpolation dose curve first became jointly eligible at alpha=0.8:

| Alpha | 8-chunk validation | 12-chunk validation | Eligible |
|---:|---:|---:|:---:|
| 0.0 | 95.12% | 91.02% | no |
| 0.4 | 96.88% | 93.95% | no |
| 0.7 | 98.24% | 94.53% | no |
| 0.8 | 98.44% | 95.51% | yes |
| 0.9 | 97.85% | 95.51% | yes |
| 1.0 | 97.66% | **96.68%** | yes, selected |

The preregistered ranking selected the fully fitted head (`alpha=1.0`). The
rescue is surgical in parameter scope, but not a tiny parameter perturbation:
the maximum absolute output-weight and output-bias changes are 0.408 and 0.585.

## Protected paired tests

| Chunks | Untouched | Head-only rescue | Change (95% CI) | Corrected / harmed |
|---:|---:|---:|---:|---:|
| 8 | 96.97% | **98.49%** | +1.51 pp `[+0.88, +2.20]` | 39 / 8 |
| 12 | 89.99% | **96.19%** | +6.20 pp `[+5.08, +7.32]` | 136 / 9 |
| 16 | 83.25% | **91.31%** | +8.06 pp `[+6.79, +9.38]` | 176 / 11 |

All paired effects are strongly positive:

- 8 chunks: McNemar `p=5.54e-6`;
- 12 chunks: McNemar `p=2.91e-30`;
- 16 chunks: McNemar `p=1.97e-39`.

The primary 12-chunk result clears 95% and its paired confidence interval is
strictly positive. The 8-chunk circuit is not merely retained; its query
accuracy improves by 1.51 points.

Local accuracy decreases by approximately one point after head replacement,
but remains high: 97.95%, 97.75%, and 97.41% at 8, 12, and 16 chunks. This is a
small shared-readout tradeoff, not loss of the local circuit.

The 16-chunk result is especially informative because no 16-chunk example was
used for training or alpha selection. The +8.06-point zero-shot improvement
suggests that a shared output-head mismatch extends across length. However,
16-chunk accuracy remains below the 95% formation target, so this result alone
does not show complete 16-chunk recovery.

## Mutation audit

Only two tensors changed:

- `output.weight`;
- `output.bias`.

The maximum change to every non-output tensor is exactly zero. Output rows
16–18 are also bit-identical to the source. The mutation audit passes.

## Rescued Memory causality

On a separate 1,024-example protected dataset:

| Condition | Query | Local |
|---|---:|---:|
| Intact | **96.29%** | 96.58% |
| Reset Memory | 5.18% | 96.58% |
| Zero Memory | 5.57% | 96.58% |
| Batch-roll Memory | 7.32% | 96.58% |

The strongest disrupted condition is 7.32%, the intact-to-disrupted drop is
88.96 points, and all causal gates pass. The rescued output head therefore
reads the same sample-specific persistent Memory circuit; it does not solve the
task through a new local shortcut.

## Scientific conclusion

For this failed seed-707 checkpoint, the 12-chunk behavioral gate conflated two
separate properties:

1. formation of a long-range Memory representation; and
2. alignment of the shared token-output head to that representation.

Level 6.18.2 showed that the representation existed. Level 6.18.3 now shows
causally that changing only the output head exposes it. Therefore the dominant
12-chunk failure mechanism is output-head misalignment, not missing persistent
Memory or failed Memory-to-token routing.

## Next falsification test

Level 6.18.4 should diagnose the residual 16-chunk gap on the same untouched
checkpoint and protected data:

1. refit frozen all-Memory and query-hidden probes at 16 chunks;
2. compare their accuracy with the original head (83.25%) and transferred
   12-chunk rescue head (91.31%);
3. quantify sample overlap among Memory decoding, query-hidden decoding, and
   the transferred head;
4. repeat the 16-chunk causal controls;
5. decide whether the remaining 3.69+ points to the 95% target are another
   length-specific output calibration problem, a Memory-to-token routing loss,
   or true Memory degradation.

Only after locating this residual should a multi-length head or renewed
formation protocol be trained. Seed 909 should remain unopened until that
protocol is frozen.

