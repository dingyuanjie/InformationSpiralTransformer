# IST v0.5 Level A formal analysis

## Verdict

The protocol completed all 15 registered runs (five variants × three seeds), but **Level A did not pass the v0.5 success gate**.

There is a genuine Evidence-path signal: Evidence-only and Hybrid exceed the empirical no-Memory/last-k baselines on average, and the strongest Hybrid seed loses its 32-chunk performance when Evidence or the Reader is zeroed. However, the result is too initialization-sensitive, the Writer loses many target relations at long distance, cross-sample Swap is not consistently harmful, and the Core path has no demonstrated benefit at 32 chunks. This is an unstable partial result, not general memory-algorithm success.

## Completion and resources

- Runs: 15/15 complete; seeds `505`, `1505`, `2505`.
- Parameter envelope: 170,563 total/trainable parameters; Memory contains 45,571 parameters.
- Total measured training time across runs: 2.61 minutes on the recorded CUDA device.
- Mean 32-chunk latency across run evaluations: 3.09 ms/example.
- Maximum recorded CUDA allocation: 21.95 MiB.
- No run reported NaN or aborted optimization.

Inactive modules remain instantiated in every variant, so total parameter envelopes match. This does not mean every variant executes identical FLOPs; inactive-path compute differences remain a disclosed limitation.

## Strict held-out accuracy

The theoretical uniform guess over 16 value tokens is 6.25%. Because held-out pairs are explicitly excluded from each entity's training support, no-Memory models can learn an anti-held-out conditional bias; their observed 0% is therefore an empirical baseline, not evidence that the task's mathematical chance is zero.

| Variant | 2 chunks | 8 chunks | 32 chunks | Interpretation |
|---|---:|---:|---:|---|
| no-Memory | 0.00% | 0.00% | 0.00% | no usable binding information |
| last-k | 0.00% | 0.00% | 0.00% | target is generally outside the tail |
| Core-only | 3.39% | 0.52% | 0.00% | abstract Core did not preserve exact bindings |
| Evidence-only | 19.01% | 11.46% | 12.76% | partial exact-evidence signal, high variance |
| Hybrid | 39.06% | 21.09% | 12.50% | highest short-range mean, extreme variance |

Hybrid standard deviations are 33.54, 25.35 and 18.37 percentage points at 2, 8 and 32 chunks. This violates the requirement that multiple seeds show a stable direction and magnitude.

## Initialization decomposition

| Hybrid seed | Last logged train accuracy | 2 chunks | 32 chunks |
|---:|---:|---:|---:|
| 505 | 21.88% | 8.59% | 0.00% |
| 1505 | 87.50% | 75.00% | 33.59% |
| 2505 | 59.38% | 33.59% | 3.91% |

Seed 1505 proves that the implemented pathway can learn a substantial held-out signal. Seed 505 proves that the present optimization/architecture does not reliably find it. Reporting only seed 1505 would be misleading.

## Writer and Reader decomposition

At 2 chunks, Hybrid Writer relation recall is 100% for every seed, while Reader hit is 72%/96%/87%. At 32 chunks, Writer recall falls to 35%/34%/31%, and Reader hit to 2%/30%/7%. Thus:

1. Short-range failure is not primarily missing evidence; it includes optimization/readout failure.
2. Long-range failure is dominated by fixed-capacity Writer competition, followed by Reader ranking.
3. Increasing training steps alone cannot repair evidence already evicted from Memory.

## Causal panel

The 32-chunk causal panel is only interpretable for seeds with nonzero normal performance.

- Hybrid seed 1505: Normal `15.63%`; Zero Evidence `0%`; Block Reader `0%`; delete target-source Chunk `9.38%`.
- Hybrid seed 2505: Normal `6.25%`; Zero Evidence `0%`; Block Reader `0%`; delete target-source Chunk `0%`.
- Zero Core leaves both seeds unchanged at `15.63%` and `6.25%` respectively. Current long-range behavior therefore depends on Evidence, not demonstrably on Core.
- Swap is not monotonic: seed 1505 changes `15.63% -> 20.31%`, while seed 2505 changes `6.25% -> 1.56%`. Small sample size and weak normal accuracy prevent a clean Swap claim.
- Shuffle is expected to be weak because attention over a set is permutation-invariant.
- The registered `corrupt_identity` implementation reverses tokens within a span. MaxSim retrieval is order-insensitive, so its null effect is structurally uninformative. This intervention must be replaced by cross-evidence identity/value reassignment before reuse.

## Scenario panel

Seed 1505 generalizes well to single-fact (`93.75%`) and moderately to multi-fact/interference/paraphrase/position variants (`56.25%`–`62.50%`). It remains weak on overwrite/temporal update (`18.75%`) and two-hop (`6.25%`). Negative queries score 0% for every seed because the training curriculum never trained an abstain/negative objective. Those scenarios are diagnostics, not passed capabilities.

## What is proved and not proved

Supported:

- exact Evidence spans can carry unseen shared-vocabulary bindings;
- the Evidence Reader can be causally necessary in a successful initialization;
- the current fixed capacity becomes a concrete long-distance bottleneck.

Not supported:

- stable multi-seed held-out generalization;
- an advantage from recursive Core State;
- reliable cross-sample identity causality;
- overwrite, temporal update, negation or two-hop competence;
- superiority over v0.1/v0.4, pretrained Transformers or real-language systems.

## Registered next step

Do not proceed to Qwen or add auxiliary losses. Run a minimal stability/retention diagnosis:

1. freeze the current task and split;
2. replace identity corruption with entity/value reassignment across evidence records;
3. separate Writer oracle (retain all target relations) from learned fixed-capacity Writer;
4. compare current hard competition against a deterministic FACT-span recency reservoir at the same capacity;
5. repeat seeds 505/1505/2505 without changing Reader/Core;
6. only if Writer-oracle still shows high seed variance, diagnose initialization and Query/Reader optimization next.

The next experiment should answer one question only: is the variance caused primarily by Writer retention or by Reader optimization?
