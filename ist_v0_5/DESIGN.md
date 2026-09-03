# IST v0.5: Hybrid Evidence–Core Recursive Memory

The state is `M_t = {E_t, C_t, P_t}`.

- `E_t`: a fixed number of exact evidence spans. Each span preserves token-level hidden states and token IDs.
- `C_t`: a small fixed recursive latent state. It integrates selected evidence but is not treated as an exact fact store.
- `P_t`: source chunk, absolute token positions, birth time, last read time, usage, importance and validity.

## Read

Query tokens and evidence tokens are projected independently. A MaxSim late-interaction score ranks evidence spans without first averaging away entity identity. Selected evidence token values are attended into the query. Core uses a separate attention path. Two learned scalar gates fuse them independently.

## Write

The Writer receives only the current chunk representation, old state and metadata. Candidate windows compete with retained evidence using content importance, novelty, age, usage and redundancy. In v0.5 Level A the inference path uses hard Top-k. A soft/straight-through writer is deliberately deferred until hard selection is shown to be the limiting factor.

## Core update

Core slots cross-attend to the newly retained evidence summary and use a gated residual LayerNorm update. `core_update="none"` and Core-only/Evidence-only variants make this mechanism independently ablatable.

## Time and provenance

Age and relative chunk distance enter read/write bias. Every returned read includes selected slots, weights, source chunks, positions and token IDs. No label is stored in metadata.

## Causal interventions

The implementation supports normal, zero/reset, Evidence-only destruction, Core-only destruction, cross-sample swap, slot shuffle, Writer block, Reader block, identity corruption, and deletion by source chunk.

## Claim boundary

The first implementation is a Level A algorithm test. Passing it means shared-vocabulary unseen bindings are learnable with fixed memory. It does not establish Qwen bridging, real-language memory, or superiority over long-context Transformers.
