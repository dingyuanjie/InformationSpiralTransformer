# Level 6.2 random-marker multi-chunk analysis

The initial constant-learning-rate run passed 2 and 4 chunks but catastrophically
collapsed at 8 chunks. Query accuracy fell from 95% at the first 8-chunk
measurement to 6.25% after 500 steps. Gate means fell to 0.094/0.195/0.150 and
first-to-final memory cosine fell to 0.393.

The stabilized run resumed from the passed 4-chunk checkpoint and used learning
rate scales 0.25 at 8 chunks and 0.10 at 16 chunks, with batch sizes 4 and 2.

| Stage | First validation | Second validation | Result |
|---|---:|---:|---:|
| 8 chunks / 1024 tokens | Query 98.75%, min probe 98.75% | Query 100%, min probe 100% | Pass |
| 16 chunks / 2048 tokens | Query 98.75%, min probe 98.75% | Query 100%, min probe 100% | Pass |

Combined with the earlier stages, Level 6.2 passed random marker retrieval across
2, 4, 8 and 16 chunks while each attention window remained 128 tokens.

## Interpretation

- Per-layer Spiral Memory can retain a randomly positioned target through 15
  intervening chunk updates when directly supervised by a memory probe.
- Long-chain optimization is sensitive to learning rate. A schedule suitable
  for 2-4 chunks destroyed memory at 8 chunks; lower rates preserved it.
- This is evidence for supervised persistent memory, not yet spontaneous memory:
  every chunk's memory state was trained to remain linearly decodable.

## Next experiment

Reduce probe weight from 0.5 to 0.2, 0.1 and 0 while resuming the passed model.
Measure the lowest probe accuracy and final query accuracy to determine whether
the memory behavior survives when direct retention supervision is withdrawn.
