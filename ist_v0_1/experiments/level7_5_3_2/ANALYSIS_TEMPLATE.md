# Level 7.5.3.2 formal analysis template

## Outcome

- Integrity: pending
- Registered classification: pending
- Runtime: pending

## Arm-level causal effects

| Arm | Material effects (of 4) | Final-fate changes | Stabilized | Destabilized |
|---|---:|---:|---:|---:|
| Reset optimizer only | pending | pending | pending | pending |
| Reset data stream only | pending | pending | pending | pending |
| Reset both | pending | pending | pending | pending |

## Outcome-stratified sources

| Seed / branch | Parent role | Exact signature | Optimizer reset | RNG reset | Both reset |
|---|---|---|---|---|---|
| 1879 / intact | Persistent L2 loss | pending | pending | pending | pending |
| 2203 / selected | Unformed L3 recovery | pending | pending | pending | pending |
| 2551 / selected | Late L3 collapse | pending | pending | pending | pending |
| 2909 / intact | Collapse then recovery | pending | pending | pending | pending |

## Interpretation boundary

This level identifies causal sensitivity to inherited optimizer state and data
order in four deliberately selected dynamics. It does not estimate population
frequencies and does not yet localize the responsible parameter group.
