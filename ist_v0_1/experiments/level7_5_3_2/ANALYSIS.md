# Level 7.5.3.2 formal analysis

## Outcome

- Integrity: PASS
- Registered classification: `optimizer_and_data_stream_both_causal`
- New training arms: 12/12
- Shared source screens: 4/4
- Trajectory screens: 48/48
- Final confirmations: 16/16
- Opposite-layer route migration: none observed
- `seed909`: excluded and locked

The experiment was outcome-stratified over four frozen endpoints. It is a
mechanism diagnostic, not a prevalence estimate over all Level 7.5.3.1
branches.

## Registered causal effects

| Intervention | Material effects | Final-fate changes | Stabilized endpoints | Destabilized endpoints |
|---|---:|---:|---:|---:|
| Reset optimizer only | 3/4 | 3/4 | 0/4 | 1/4 |
| Reset data stream only | 4/4 | 1/4 | 0/4 | 1/4 |
| Reset both | 4/4 | 1/4 | 0/4 | 1/4 |

The optimizer-only intervention changed the final route fate in three of the
four diagnostic endpoints. The data-stream-only intervention changed every
registered trajectory materially, even when the final route class was retained.
The joint reset also changed every trajectory. Therefore neither inherited
AdamW state nor inherited stochastic order can be treated as incidental; both
are causal contributors to the observed volatility.

## Endpoint summary

| Endpoint | Exact reference | Reset optimizer | Reset data stream | Reset both |
|---|---|---|---|---|
| seed1879 intact | unformed, 78.9% | unformed, 76.6% | unformed, 86.5% | unformed, 77.3% |
| seed2203 selected | L3 core, 96.5% | unformed, 77.0% | unformed, 80.0% | unformed, 82.0% |
| seed2551 selected | unformed, 25.6% | L3 core, 94.8% | unformed, 73.2% | unformed, 6.4% |
| seed2909 intact | L3 core, 93.8% | unformed, 82.4% | L3 core, 95.4% | L3 core, 92.4% |

The interventions do not simply “repair” the system. For example, resetting
the optimizer destroys the seed2203 recovery but rescues the seed2551 late
collapse; resetting the data stream removes the seed2203 recovery while
retaining the final seed2909 route. The same factor can therefore move the
system in opposite directions depending on the endpoint basin and phase.

## Interpretation

Level 7.5.3.1 identified route formation/collapse as metastable and
path-dependent. Level 7.5.3.2 adds that the path is jointly controlled by the
optimizer's recurrent state and the stochastic training stream. Endpoint
weights alone do not determine the next 1,000-step trajectory. No evidence
supports an L2-to-L3 or L3-to-L2 replacement route; the interventions change
formation, collapse timing, and recovery within the source topology.

This result does not isolate a parameter group or prove a population-level
frequency. The next causal stage should freeze the optimizer/data-stream
factorial and intervene on parameter groups or Memory update gates while
holding both factors fixed.

## Reproduction artifacts

- `result.json`: complete raw formal result
- `summary.json`: compact diagnosis and endpoint table
- `optimizer_rng_bifurcation.png`: trajectory visualization
- `preregistration.json`: locked protocol and hashes
- `progress.json`: completion and integrity state
