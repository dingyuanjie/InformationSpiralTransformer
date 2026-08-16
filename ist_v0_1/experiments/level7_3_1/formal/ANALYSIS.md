# Level 7.3.1 formal analysis

## Decision

- Run integrity: **PASS**.
- Frozen checkpoint: seed1879 zero-Probe step750, exactly as registered.
- Fixed fresh panel: **8,192/8,192 samples** for every condition.
- Registered conditions: **11/11 complete**.
- Registered gates: **8/9 pass**.
- Keep-L2 single-layer sufficiency gate: **FAIL**.
- Registered classification:
  `l2_route_supported_but_single_layer_sufficiency_inconclusive`.

The run completed normally in 1,450.33 seconds (24.17 minutes). The
classification is a statistical result, not a program error or incomplete
resume.

## Frozen source and integrity

The only model was the preregistered seed1879 checkpoint selected and tested in
Level 7.2, then classified as L2 dominant in Level 7.3:

- checkpoint:
  `experiments/level7_2/formal/seed1879/zero_probe_step0750.pt`;
- checkpoint SHA-256:
  `cbba0c6db219e16be274bd0e77612972b7ad7d24ffa2bddb7b7cb858bf7a74f6`;
- model fingerprint before and after all interventions:
  `ad113264a0227bb69770dc56b7f395b0bf999e7b4eb8c487408d07100043f77c`.

The fingerprint was unchanged, all parameters were frozen, all conditions
were evaluated at the fixed N=8192 endpoint, and condition order was exact.
No model, Probe, checkpoint, output head, or router was trained or selected.
The new dataset seed 7,310,000 differs from the Level 7.2 and Level 7.3 panel
seeds. Seed909 remained closed.

The recorded source hashes match the current files:

- runner SHA-256:
  `d8711e006a8ea08d241cc322838feaa24d75c71f04aed295ea5b76bd1508db9e`;
- static preregistration SHA-256:
  `8eaa74411948b7d3d05b7c03b78ab79db71fa4aba3cbe041f861da324049b7f4`.

Aggregate integrity: **PASS**.

## Fixed high-precision panel

| Condition | Correct / N | Query | 95% Wilson interval | Local |
|---|---:|---:|---:|---:|
| Intact | 7,917 / 8,192 | 96.6431% | [96.2306%, 97.0118%] | 99.6948% |
| Reset all | 500 / 8,192 | 6.1035% | [5.6054%, 6.6428%] | 99.6948% |
| Zero all | 500 / 8,192 | 6.1035% | [5.6054%, 6.6428%] | 99.6948% |
| Batch-roll all | 528 / 8,192 | 6.4453% | [5.9337%, 6.9977%] | 99.6948% |
| Zero L2 | 1,055 / 8,192 | 12.8784% | [12.1704%, 13.6212%] | 99.6948% |
| Batch-roll L2 | 1,060 / 8,192 | 12.9395% | [12.2300%, 13.6837%] | 99.6948% |
| Keep L2 only | 7,401 / 8,192 | **90.3442%** | **[89.6856%, 90.9650%]** | 99.6948% |
| Zero L3 | 7,396 / 8,192 | 90.2832% | [89.6228%, 90.9058%] | 99.6948% |
| Batch-roll L3 | 6,949 / 8,192 | 84.8267% | [84.0335%, 85.5872%] | 99.6948% |
| Keep L3 only | 1,036 / 8,192 | 12.6465% | [11.9442%, 13.3838%] | 99.6948% |
| Keep L2+L3 | 7,928 / 8,192 | **96.7773%** | **[96.3725%, 97.1384%]** | 99.6948% |

Local behavior and its confidence interval are identical across conditions
because the intervention begins after the first chunk. The local Wilson lower
bound is 99.5499%, safely above the registered 90% gate.

## Preregistered confidence gates

| Gate | Registered rule | Observed decision value | Result |
|---|---|---:|---|
| Intact behavior | intact lower >=90% | 96.2306% | PASS |
| Local behavior | minimum local lower >=90% | 99.5499% | PASS |
| Whole-Memory causality | all disruption uppers <=20% | maximum 6.9977% | PASS |
| L2 necessity | zero-L2 upper <=20% | 13.6212% | PASS |
| L2 sample alignment | roll-L2 upper <=20% | 13.6837% | PASS |
| L2 single-layer sufficiency | keep-L2 lower >=90% | **89.6856%** | **FAIL** |
| L3 nonnecessity contrast | zero/roll-L3 lowers >=80% | minimum 84.0335% | PASS |
| L3 insufficiency contrast | keep-L3 upper <=20% | 13.3838% | PASS |
| L2+L3 positive control | keep-L2-L3 lower >=90% | 96.3725% | PASS |

The sole failed gate is exactly the uncertainty that motivated this stage.
Keep-L2 remains above 90% as a point estimate, but its 95% lower confidence
bound does not reach 90%. The strongest preregistered classification therefore
cannot be awarded.

## Replication against Level 7.3

The larger independent panel closely reproduces the earlier causal profile:

| Condition | Level 7.3 N=2,048 | Level 7.3.1 N=8,192 | Change |
|---|---:|---:|---:|
| Keep L2 | 90.9180% | 90.3442% | -0.5737 pp |
| Zero L2 | 12.9395% | 12.8784% | -0.0610 pp |
| Batch-roll L2 | 11.6211% | 12.9395% | +1.3184 pp |
| Zero L3 | 90.3809% | 90.2832% | -0.0977 pp |
| Batch-roll L3 | 85.2051% | 84.8267% | -0.3784 pp |
| Keep L3 | 13.3789% | 12.6465% | -0.7324 pp |
| Keep L2+L3 | 97.0703% | 96.7773% | -0.2930 pp |

Thus the precision failure is not caused by a route reversal. The key effects
replicate: L2 destruction remains catastrophic, L3 alone remains inadequate,
and L2+L3 remains sufficient at nearly intact performance. The larger panel
narrows the uncertainty and shows that the keep-L2 effect sits too close to
90% for the registered lower-bound claim.

## Mechanistic interpretation

The result sharpens the phrase "L2 dominant":

1. **L2 is the indispensable core route.** Zeroing or sample-mismatching L2
   reduces query accuracy to 12.88-12.94%, and the upper confidence bounds are
   below 13.69%.
2. **L2 alone carries most, but not robustly all, deployed performance.** It
   reaches 90.34%, but not the registered high-precision sufficiency standard.
3. **L3 is supportive rather than irrelevant.** L3 alone reaches only 12.65%,
   yet keeping correct L3 alongside L2 raises behavior from 90.34% to 96.78%.
   Batch-rolling L3 also reduces the otherwise intact route to 84.83%, showing
   that L3 contains useful sample-specific state.
4. **The L2+L3 route is robustly sufficient.** Its Wilson lower bound is
   96.37%, and it matches intact performance without L1.
5. **Complete Memory causality replicates again.** Reset, zero, and batch-roll
   controls remain near chance while local behavior remains solved.

The best description of seed1879 is therefore an **L2-core, L3-supported
distributed circuit**, not an L2-only circuit. This remains qualitatively
different from the sharply L3-sufficient circuits in seeds 606, 808, and 1001.

## Scientific conclusion

Level 7.3.1 strengthens the cross-initialization heterogeneity conclusion while
narrowing one layer-specific claim:

- L2 necessity and sample alignment are independently replicated with narrow
  confidence intervals;
- isolated L3 is decisively insufficient;
- isolated L2 is behaviorally strong but does not satisfy the registered
  high-precision >=90% lower-bound definition of sufficiency;
- L2+L3 is robustly sufficient and nearly identical to intact behavior;
- the formal Level 7.3.1 result is therefore supportive but not the strongest
  registered confirmation.

This frozen-checkpoint study does not address training reliability, route
prevalence across new initializations, or comparison with a standard
Transformer. The Level 7.1 formation-reliability limitation remains unchanged.

## Registered stop boundary and next study

Do not append samples to this panel or rerun it with changed thresholds. Level
7.3.1 is closed as
`l2_route_supported_but_single_layer_sufficiency_inconclusive`. Adding more
samples until the lower bound crosses 90% would violate the fixed-N protocol
and turn precision testing into optional stopping.

The next mechanistic study should move away from the boundary chase. A useful
separately preregistered phase would test how the L2-core/L3-support route
emerges and stabilizes across fixed withdrawal checkpoints, or evaluate route
classes in new untouched initializations. Either would answer a new question;
neither should relabel this result.
