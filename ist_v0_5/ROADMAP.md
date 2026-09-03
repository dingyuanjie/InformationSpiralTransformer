# Staged delivery status

## Implemented now

- repository/v0.4 audit and two-candidate design comparison;
- minimal multi-vector Evidence + recursive Core architecture;
- strict shared-vocabulary/unseen-binding generator and leakage checker;
- Level A no-memory, parameter-envelope-matched, last-k, Core-only, Evidence-only and hybrid runs;
- distance, density and registered scenario evaluation;
- causal intervention panel and source provenance;
- parameter, token, time, latency, peak-memory and Memory-health metrics;
- three-seed formal configuration and automatic JSON/CSV/Markdown/PNG reports;
- failure classification, tests and smoke result.

## Gated, not yet implemented or claimed

- v0.1/v0.4 adapters under the new exact Level A protocol;
- semi-natural Chinese/English Level B;
- frozen Qwen2.5-0.5B Level C;
- soft versus straight-through Writer ablation;
- residual versus GRU Core ablation;
- layer-placement experiments, LoRA, partial unfreezing and 1B scaling.

These items are intentionally gated. Per protocol, they should not be added until the formal Level A run shows a stable held-out signal. This prevents a negative minimal result from being hidden under extra modules or compute.

## Formal Level A decision

The first formal run failed the stability gate. v0.5.1 now implements the registered Writer-oracle versus fixed-capacity retention diagnosis, corrected binding corruption, five-seed Reader stability comparison and capacity curve. Level B and Level C remain closed until its formal result passes.

v0.5.1 passed its Reader stability gate. The next registered stage is v0.5.1.1: component ablation of the stabilization package and distractor-count scaling under 100% target availability. Level B/C remain closed until the minimal causal Reader mechanism is isolated.
