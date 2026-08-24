# Level 8.1 Formal Analysis

## Integrity

- Complete: 2 architectures x 5 seeds x 2 evaluation replicates.
- 160 paired samples per distance, at 1--1000 chunks.
- Both architectures remain statistically above the 1/16 chance level at 1000 chunks.

## Primary result

The hierarchical v0.2 implementation is functional and retains task information to 1000 chunks, but it does **not** improve retention over v0.1 under the matched protocol.

| chunks | v0.1 | hierarchical v0.2 | paired delta | McNemar p |
|---:|---:|---:|---:|---:|
| 128 | 66.88% | 64.38% | -2.50 pp | 0.388 |
| 256 | 61.25% | 55.00% | -6.25 pp | 0.099 |
| 512 | 60.00% | 42.50% | -17.50 pp | 0.000008 |
| 1000 | 44.38% | 34.38% | -10.00 pp | 0.009 |

At 512 and 1000 chunks the regression is statistically significant. The nominal lifetime metric saturates at the protocol ceiling (1000) for both models, so it cannot distinguish them; the accuracy curve and paired tests are the meaningful endpoints.

## Mechanistic indication

- Layer 2 routes almost entirely to Fast Memory by long distances (about 99.7% at 1000 chunks), while Slow and Episodic writes approach zero.
- Slow Memory remains very similar to initialization (roughly 0.96--0.97 in the displayed long-distance diagnostics), consistent with under-use rather than useful consolidation.
- Episodic target similarity is weak or negative in several layers. Finite capacity alone therefore did not create selective durable storage.
- Seed variation is substantial, but the long-distance deficit is not produced by a single evaluation replicate. Seeds 7 and 1234 show clear degradation; seed 313 is at chance for both; seed 2026 remains strong for both.

## Decision

Do not claim that v0.2 beats v0.1. Proceed to Level 8.2 as a causal component/router diagnosis before tuning capacity or adding parameters. The first targets should be Router collapse, Slow consolidation inactivity, and whether Episodic writes carry query-causal information.
