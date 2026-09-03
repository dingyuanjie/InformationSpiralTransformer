# IST v0.5.1 formal analysis

## Verdict

v0.5.1 completed all 10 registered Oracle Reader runs (two conditions × five seeds) and **passed the Reader stability gate**. It also changes the diagnosis of the original 32-chunk result:

1. the real Writer's exact-occurrence retention follows the query-blind fixed-capacity ceiling `K/64` almost exactly;
2. the earlier 31%–35% relation recall was inflated by alternate copies of the same binding and was not exact-source retention;
3. with exact target Evidence forced into Memory, Reader accuracy is stable across all five seeds;
4. increasing capacity retains more evidence but creates a larger ranking problem, so capacity alone does not yield perfect accuracy;
5. the Core path remains outside this experiment and gains no new support here.

This is a positive mechanism result for Evidence Memory and Reader stabilization, not yet a complete v0.5 architecture success.

## Oracle Reader stability

| Condition | 2 chunks | 4 chunks | 8 chunks | 16 chunks | 32 chunks |
|---|---:|---:|---:|---:|---:|
| Oracle current | 93% ± 7% | 91% ± 7% | 87% ± 8% | 83% ± 4% | 80% ± 5% |
| Oracle stable | **100% ± 0%** | **100% ± 0%** | **100% ± 0%** | **95% ± 1%** | **90% ± 4%** |

Every stable seed reaches 100% training accuracy. At 32 chunks their held-out accuracies are `87%`, `95%`, `85%`, `92%` and `91%`. Exact target retention is 100% by Oracle construction, and Reader exact-source hit is `99%–100%` for all five stable seeds.

The stabilization package (open Evidence gate, closed Core gate, temperature 0.7, order-sensitive reranker and retrieval loss) therefore removes the seed failure seen in v0.5 Level A. This experiment does not isolate which member of that package is necessary; a component ablation is still required before attributing the gain.

## Capacity ceiling audit

The 32-chunk stream contains 64 fact occurrences and the Writer cannot see the future query. For equally queryable facts, exact target retention is bounded in expectation by `K/64`.

Stable Reader means across five seeds:

| K | K/64 ceiling | Exact source retained | Same binding retained | Accuracy | Accuracy given exact retained | Accuracy given exact absent | Reader exact hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.25% | 6% | 14% | 18% | 100% | 13% | 6% |
| 8 | 12.50% | 12% | 25% | 26% | 93% | 17% | 12% |
| 12 | 18.75% | 18% | 34% | 33% | 89% | 21% | 17% |
| 16 | 25.00% | 24% | 41% | 38% | 87% | 23% | 24% |
| 24 | 37.50% | 39% | 54% | 45% | 78% | 23% | 38% |
| 32 | 50.00% | 51% | 65% | 52% | 76% | 26% | 47% |
| 64 | 100.00% | 100% | 100% | 61% | 61% | — | 73% |

Exact retention tracks the information-capacity line within sampling error. The Writer is therefore not showing an obvious selection deficit on this exchangeable random-fact task. Tuning age or importance cannot beat the average query-blind ceiling without exploitable priority information.

Accuracy can exceed exact-source retention because another occurrence of the same binding may remain, and because the trained model has residual binding/generalization signal. This validates the decision to report exact occurrence and same binding separately.

At K=64 every exact target survives, but accuracy is only 61% and exact Reader hit 73%. This exposes a second scaling law: larger storage reduces eviction but increases retrieval distractors. The next optimization target is candidate ranking under growing evidence count, not unconditional storage expansion.

## Binding causality

Under Oracle Evidence, original mean accuracy is 87% for the current Reader and 100% for the stable Reader at the registered causal distance.

Stable Reader results:

| Intervention | Prediction changed | Followed counterfactual binding |
|---|---:|---:|
| swap entities | 75% | 53% |
| swap answers | 90% | 68% |
| rebind both | 88% | 57% |
| corrupt roles | 27% | not defined by current evaluator |

Predictions frequently follow the newly assigned answer rather than merely becoming wrong, providing causal evidence that the Reader uses entity–answer binding content. The follow rate is not yet close to 100%, so role-sensitive reranking remains incomplete. `corrupt_roles` destroys the evaluator's entity-at-role-1 lookup and its follow metric is intentionally not interpreted.

## Corrected decomposition

The tested pipeline is now best described as:

`accuracy ≈ exact retention × conditional read accuracy + alternate-binding/absent contribution`.

- At K=12: exact retention ≈18%, conditional accuracy ≈89%, absent accuracy ≈21%, total ≈33%.
- At K=64: exact retention =100%, but conditional accuracy falls to61% because the Reader sees many more distractors.

Thus the original seed variance was substantially a Reader optimization problem, while the original long-distance ceiling was a mixture of finite storage and ranking under interference. The exact Writer itself is approximately capacity-optimal for exchangeable facts.

## Claim boundary

Supported:

- five-seed stable Reader learning when the exact target is available;
- predictable fixed-capacity retention at the query-blind information ceiling;
- causal sensitivity to entity/answer rebinding;
- a measurable storage-versus-retrieval-interference tradeoff.

Not supported:

- selective retention above `K/N` without a predictive importance signal;
- perfect retrieval when capacity grows;
- an independent contribution from Core State;
- overwrite, two-hop, negation, natural language or pretrained-Backbone success;
- superiority over Transformer or other memory architectures.

## Registered next step

The next minimal stage should be **v0.5.1.1 Reader stabilization ablation and distractor scaling**:

1. ablate gate initialization, Core closure, temperature, reranker and contrastive loss one at a time;
2. hold exact target availability at 100%;
3. vary evidence count independently of distance;
4. report top-1/top-3/top-k target recall and accuracy;
5. select the smallest stabilization subset that succeeds in all five seeds;
6. only then reconnect the real Writer and test capacity-aware retrieval.

Do not proceed to Qwen or revive Core on the random-binding task yet.
