# IST v0.5.1 protocol

## Question

Is cross-seed variance caused primarily by target evidence eviction or by Reader optimization when the exact target is available?

## Registered conditions

- `oracle_current`: exact target occurrence forced into Evidence; original 0.5 Evidence gate; MaxSim only; task loss only.
- `oracle_stable`: exact target forced; Evidence gate starts at sigmoid(2)=0.881; Core gate is closed; temperature is 0.7; order-sensitive reranker and 0.2 retrieval contrastive loss are enabled.

Both conditions use Evidence-only output and five seeds. No Qwen, Core claim, Writer auxiliary loss or future Query enters ordinary historical write decisions. Oracle insertion is explicitly supervised and diagnostic.

## Capacity audit

Each trained Reader is evaluated with the real query-blind Writer at capacities 4, 8, 12, 16, 24, 32 and 64 on a 32-chunk stream containing two facts per chunk. For exact fact occurrences, the uniform-retention ceiling is `min(1, K/64)`. Metrics distinguish:

- exact source occurrence retained;
- any duplicate of the same entity–answer binding retained;
- accuracy given exact retention;
- accuracy given exact absence;
- Reader exact-source hit.

## Binding causality

Entity tokens, answer tokens, both sides of the binding, or role positions are changed while retaining the global token multiset where applicable. A causal claim requires predictions to follow the counterfactual binding, not merely change.

## Gate

Reader stabilization passes only if all five seeds are above the 6.25% value-guess rate at trained distances, at least three remain above it under length extrapolation, `accuracy_given_exact_retained` is high, and Evidence/Reader intervention effects are directionally consistent. Failure keeps Level B and Qwen closed.
