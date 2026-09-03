# IST v0.5 pre-implementation audit

Audit date: 2026-09-04. Evidence source: tracked repository code and result files through commit `89eb3cf`.

## Repository and version state

- No `AGENTS.md` exists and the worktree was clean at audit start.
- v0.1 is the synthetic recurrent-slot branch. Its frozen evidence ledger supports persistent and causally usable Memory on the tested task, while recording failed router repair.
- v0.2 is the hierarchical adapter branch. Its final frozen interventions did not beat baseline.
- v0.3 stores source token states and provenance. Writer coverage passed, but strict OOD retrieval and decoding failed.
- The repository contains `ist_v0_4`, not a separate `ist_v0_4_1`. “v0.4.1” therefore has no independently reproducible code or result identity.
- v0.4 implements working, episodic and semantic event memory. The latest relational Query run completed, but held-out Top-1 stayed at the 25% four-way chance level.

## v0.4 data flow

1. `FrozenCognitiveIST` installs a pre-hook at Qwen layer `-4`.
2. If old state exists, normalized current hidden states query working/episodic/semantic Memory and gated context is added before that decoder layer.
3. The pre-injection hidden state is captured with `detach()`.
4. After the Backbone forward pass, overlapping 24-token/8-stride events are built from the captured state.
5. Event mean states are projected to keys. Surprise, novelty and redundancy form admission scores.
6. Working Memory is a recency FIFO. Episodic entries compete using strength, age, idle time and access count. Repeatedly accessed entries can update semantic prototypes.
7. Formal Query alignment writes historical chunks under `torch.no_grad()`, calls `detach_state=True` after every chunk, freezes Qwen/Writer/Value/Output/lifecycle, and trains only Query, event Key and Query LayerNorm.
8. The later query is not visible when historical facts are written. It only affects the read. There is no cross-chunk gradient through stored historical states in this protocol.

## Leakage and fairness audit

- Training and held-out entity strings, answer token IDs and query templates are disjoint in the latest v0.4 run.
- Facts are placed from deterministic but separate seed ranges. The query is not present during history writing.
- A fixed token-label map is avoided by resampling bindings per example.
- Important confound: held-out answer IDs are excluded from Query training. This tests open-token representation and Memory lookup together, and can turn an output/embedding limitation into an apparent Memory failure.
- Important weakness: only one formal seed is reported for latest v0.4 Query alignment.
- Important weakness: no parameter/compute-matched no-Memory, last-k or GRU baseline is included in that run.
- Important weakness: validation uses one held-out query template and four held-out entities, so it is strict but statistically narrow.
- The current protocol reports Writer availability and retrieval separately, which correctly prevents a missing write from being counted as a Reader failure.

## Most likely causes of held-out failure

1. **Representation bottleneck.** A complete 24-token relation is mean-pooled to one key, diluting entity identity and local binding.
2. **Query bottleneck.** Retrieval supervision uses only the final query-token score; a frozen intermediate Qwen state may not carry enough entity identity at that position.
3. **Conflated generalization axes.** New bindings, new entity strings, new query wording and untrained answer tokens change simultaneously. v0.3 already showed open-answer decoding can fail even under an Oracle Reader.

## Two minimal candidates

### Candidate A — multi-vector Evidence plus residual Core (selected)

Store short evidence spans as token-level hidden states with token IDs, positions, source chunk and lifecycle metadata. Retrieve evidence with late token interaction and independently read a small recursively updated Core. This directly addresses identity dilution and supports source deletion interventions. Cost is `O(query_tokens × evidence_capacity × span)` with small fixed capacities.

### Candidate B — pooled span Evidence plus GRU Core

Store one attention-pooled vector per span and update Core with a GRU. It is cheaper, but remains vulnerable to compressing entity and value into an anonymous vector. It is retained as a future ablation, not the first implementation.

## Selected minimum and experiment order

Implement Candidate A with hard fixed-capacity competition, residual gated Core update, shared-vocabulary/new-binding splits, and independent Evidence/Core read gates. Start with task loss only. Level A first compares no-memory, last-k, Core-only, Evidence-only and hybrid on equal examples and seeds. Causal interventions are only interpreted after held-out performance exceeds chance.

This audit does not authorize a claim of natural-language or universal Transformer superiority.
