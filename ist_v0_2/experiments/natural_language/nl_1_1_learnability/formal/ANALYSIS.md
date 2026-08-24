# NL-1.1 Formal Analysis

## Integrity

- Complete: three architectures x five seeds (15 runs).
- 320 examples per architecture/split/distance at 512, 1K, and 2K.
- All stored data audits passed. Four-way chance is 25%.

## Held-out result

| distance | Transformer | IST v0.1 | IST v0.2 |
|---:|---:|---:|---:|
| 512 | 28.75% | 30.63% | 30.63% |
| 1K | 23.44% | 25.31% | 25.31% |
| 2K | 25.00% | 27.19% | 27.19% |

Only the two IST rows at 512 have a Wilson lower bound above chance (about 25.8%). Neither IST is above chance at 1K, so the declared learnability gate fails.

## Shortcut diagnosis

The 512 result is not accepted as semantic task acquisition. IST v0.1 and v0.2 predictions agree on 100% of held-out examples at every distance. Both predict only answer ids 258 and 260, while the targets are well distributed over all four answer ids. At each distance their prediction histogram is exactly 256 predictions of id 260 and 64 of id 258. This is consistent with a shared shallow template/category rule rather than locating and matching the earlier natural-language fact.

Training also remained unstable: final stage mini-batch accuracy varied from 0% to 50%, and the loss did not demonstrate a clean task solution.

## Decision

NL-1.1 fails its gate. Do not interpret the 512 bump as Persistent Memory, do not proceed to NL-2, and do not claim synthetic-to-natural transfer. The next diagnostic should decompose acquisition into (A) answer-token decoding, (B) exact lexical option matching, and (C) paraphrased semantic matching. This will locate whether failure is in the output protocol, byte-level comparison, or language semantics before deciding whether a pretrained base is required.
