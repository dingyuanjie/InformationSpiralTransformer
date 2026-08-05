# Level 6 cross-chunk persistent-memory analysis

Chance accuracy is 6.25%. All variants remained near chance at every chunk
count, so the current experiment does not demonstrate cross-chunk memory.

| Variant | 2 chunks | 4 chunks | 8 chunks | 16 chunks | Time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 7.5% | 12.5% | 5.0% | 2.5% | 97.0 s | 88.2 MB |
| IST-Persistent | 5.0% | 2.5% | 7.5% | 5.0% | 312.4 s | 153.3 MB |
| IST-Reset | 5.0% | 2.5% | 7.5% | 5.0% | 185.8 s | 155.1 MB |

Persistent and reset IST produced the same aggregate accuracies and nearly
identical learning curves. The persistent path therefore supplied no measurable
task information under this protocol.

## Likely causes

1. The current model carries one memory tensor through all layers. The final
   layer's memory is then fed into the first layer of the next chunk, mixing
   representation depths.
2. No objective directly requires the first chunk's target to be decodable from
   memory. Slot compression may simply discard the marked token.
3. Memory is repeatedly updated by noise chunks, so useful content can be
   overwritten before the query arrives.
4. Backpropagation across up to 16 chunks is expensive and provides a weak,
   distant learning signal. Persistent IST was 3.2x slower than Transformer.

## Required Level 6.1 diagnostic

- Use two 128-token chunks and a fixed marker first.
- Maintain one persistent memory tensor per Transformer layer so each layer
  updates its own state across chunks.
- Add a memory probe head and auxiliary loss requiring the target to be decoded
  from memory after chunk 1 and retained after chunk 2.
- Report probe accuracy, final query accuracy, gate values and memory similarity
  before attempting 4/8/16 chunks.

Only after the two-chunk probe and query both exceed 95% should the curriculum
expand to 512-token chunks and 8192 total tokens.
