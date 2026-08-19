# Level 7.5.3.3 formal analysis

## Outcome

- Integrity: PASS
- Registered classification: `distributed_l2_l3_memory_pathway`
- New training arms: 16/16
- Shared source screens: 4/4
- Trajectory screens: 60/60
- Final confirmations: 20/20
- Frozen parameter snapshots unchanged: PASS
- AdamW and CPU/CUDA RNG preservation: PASS
- Opposite-layer migration: none observed
- `seed909`: excluded and locked

This is a mechanism diagnostic over four outcome-stratified endpoints, not a
prevalence estimate over the full seed set.

## Registered effects

| Intervention | Material effects | Gate/pathway comparison |
|---|---:|---:|
| Freeze L2 Memory pathway | 3/4 | L2 gate matched 2/4 |
| Freeze L3 Memory pathway | 4/4 | L3 gate matched 4/4 |
| Freeze L2 update gate | 3/4 | — |
| Freeze L3 update gate | 4/4 | — |

The registered classification is distributed across L2 and L3 Memory pathways.
The result does not support a single-layer or single-gate explanation.

## Endpoint interpretation

- Freezing the L2 Memory pathway rescued the persistent L2-loss endpoint
  (`seed1879`) to a final 93.9% expected L2 route, but did not rescue the
  L3-diagnostic endpoints.
- Freezing the L3 Memory pathway prevented the `seed2203` L3 recovery,
  prevented the `seed2551` late collapse, and removed the final `seed2909`
  L3 route. Thus the same L3 pathway controls both formation and collapse,
  depending on the endpoint phase.
- Freezing the L3 update gate reproduced the registered material-effect status
  of the full L3 pathway in all four diagnostic comparisons, but it did not
  meet the preregistered L3-gate-dominant outcome because L2 pathway effects
  were also present.
- Freezing the L2 update gate caused a catastrophic `seed2909` collapse
  (6.5% final query) and materially changed two other trajectories, showing
  cross-layer coupling rather than an isolated L2-only function.

The dominant retention layer stayed within the source topology. The effects are
therefore formation, collapse, and recovery changes inside a distributed L2/L3
Memory system, not L2↔L3 route replacement.

## Conclusion

With optimizer state and data stream fixed, Memory parameter updates themselves
are causally responsible for the metastable route dynamics. L3 is the strongest
direct control point for the selected L3 endpoints, but L2 and both update gates
can modulate the same trajectories. The next experiment should use a finer
layer-by-layer gate/slot intervention or frozen-gate readout test; it should not
be interpreted as evidence that one universal layer owns all long-context
Memory.

## Reproduction artifacts

- `result.json`: complete raw formal result
- `summary.json`: compact diagnosis and comparisons
- `memory_parameter_group_effects.png`: trajectory visualization
- `preregistration.json`: locked protocol
