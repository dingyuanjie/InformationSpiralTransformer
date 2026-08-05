# Level 6.1 minimal cross-chunk diagnostic

The diagnostic passed after two consecutive validations at steps 1300 and 1400.

| Metric | Final accuracy |
|---|---:|
| Local target prediction | 100% |
| Memory probe after chunk 1 | 100% |
| Memory probe after chunk 2 | 100% |
| Query in chunk 2 | 100% |

Final mean memory cosine similarity between chunk 1 and chunk 2 was 0.659.
Mean update gates were 0.440, 0.440 and 0.398 for layers 1-3. The lower-than-0.5
gates indicate a learned bias toward retaining prior memory while processing the
second noise/query chunk.

## What this establishes

- Per-layer persistent memories can carry target information across chunks.
- The target remains linearly decodable after both the write chunk and the
  following noise/query chunk.
- The final query can read the carried state; there is no direct token path from
  chunk 1 to chunk 2 other than the per-layer memory list.

## What this does not establish

- The marker was fixed at the start of chunk 1.
- A direct memory-probe auxiliary objective explicitly trained writing and
  retention.
- Only two 128-token chunks were tested.
- This result does not yet show retention through many noise chunks or without
  probe supervision.

## Next diagnostic

Randomize the marker position, retain the probe loss, and expand through 2, 4,
8 and 16 chunks while keeping per-layer memory. Once stable, reduce the probe
weight to test whether the query loss alone can maintain the memory behavior.
