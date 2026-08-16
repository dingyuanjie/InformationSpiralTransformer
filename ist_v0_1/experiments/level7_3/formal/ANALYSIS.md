# Level 7.3 formal analysis

## Decision

- Run integrity: **PASS**.
- Frozen checkpoints completed: **4/4** (seeds 606, 808, 1001, and 1879).
- Registered conditions completed: **16/16 for every checkpoint**.
- Whole persistent-Memory causal gate: **PASS for 4/4 checkpoints**.
- Exact registered layer signatures: **3**.
- Dominant layer classes: **2** (layer 3 for seeds 606/808/1001 and layer 2
  for seed1879).
- Registered classification:
  `cross_initialization_layer_heterogeneity_confirmed`.

This is a completed positive mechanistic result. It is not a training result:
all four models were frozen before the shared causal panel was opened.

## Integrity and frozen sources

The experiment used the four preregistered source checkpoints and the single
fresh dataset seed 7,300,000. Every condition used the same 2,048 examples.
No Probe, model parameter, output head, or router was trained or selected.
Seed909 remained closed.

| Seed | Source behavior | Checkpoint SHA-256 |
|---:|---:|---|
| 606 | 97.7500% | `488f354fd47eab511fbc0d29697fec8edd54400c666d083717bb008537896ca5` |
| 808 | 96.2500% | `bd490973879328accf67b1abdfe5ddd76a7f3dc86bebf507694c772b972c9a84` |
| 1001 | 97.2500% | `0cf83448dbd229e3796b08feb120b0b1c5cc4df25907b156cd7a5905899a13d1` |
| 1879 | 97.0947% | `cbba0c6db219e16be274bd0e77612972b7ad7d24ffa2bddb7b7cb858bf7a74f6` |

Each model fingerprint was identical before and after all 16 interventions,
all parameters were frozen, and all condition records were present. Aggregate
integrity is **PASS**. The recorded runner and static preregistration hashes
are:

- runner SHA-256:
  `b0e20fd4effe56b2ea195423c6cb17ee3e575444a2287f2a52e3b80ad9475ba8`;
- static preregistration SHA-256:
  `a5afc29920d87398c8684b7f1b6bdd8c798de807fc1c0637f4f3d8fce1620855`.

## Shared 16-chunk causal atlas

The table reports query accuracy on the common fresh panel.

| Condition | Seed606 | Seed808 | Seed1001 | Seed1879 |
|---|---:|---:|---:|---:|
| Intact | 96.14% | 95.65% | 97.02% | 96.34% |
| Reset all | 6.88% | 5.71% | 6.10% | 7.13% |
| Zero all | 6.88% | 6.15% | 6.01% | 7.13% |
| Batch-roll all | 5.62% | 5.62% | 5.62% | 5.52% |
| Zero L1 | 96.34% | 98.39% | 97.66% | 97.07% |
| Zero L2 | 97.22% | 80.32% | 90.87% | **12.94%** |
| Zero L3 | **6.74%** | **6.05%** | **6.10%** | 90.38% |
| Batch-roll L1 | 96.14% | 95.85% | 97.66% | 96.53% |
| Batch-roll L2 | 96.68% | 95.85% | 97.71% | **11.62%** |
| Batch-roll L3 | **5.47%** | **5.37%** | **5.66%** | 85.21% |
| Keep L1 only | 6.88% | 5.81% | 6.30% | 7.03% |
| Keep L2 only | 6.79% | 6.30% | 5.76% | **90.92%** |
| Keep L3 only | **97.17%** | **98.49%** | **98.68%** | 13.38% |
| Keep L1+L2 | 6.74% | 6.05% | 6.10% | 90.38% |
| Keep L1+L3 | 97.22% | 80.32% | 90.87% | 12.94% |
| Keep L2+L3 | 96.34% | 98.39% | 97.66% | 97.07% |

Local accuracy remained between 97.90% and 99.61% for all interventions in a
given model. The causal collapse is therefore selective to cross-chunk query
behavior rather than a general failure of local token processing.

## Whole-Memory causal replication

All four intact query scores were above the registered 90% threshold. Reset,
zero, and batch-roll of all Memory reduced every model to 5.52-7.13%, far
below the registered 20% disruption threshold and close to chance. Their 95%
Wilson upper bounds were at most 8.33%.

Consequently, complete persistent Memory is causally necessary and
sample-specific in **4/4 successfully formed checkpoints**. This result is
stronger than a decodability claim: replacing the Memory with another
sample's state destroys behavior while leaving local performance intact.

## Registered layer signatures

| Seed | Necessary | Misassignment-sensitive | Sufficient alone | Sufficient pairs | Class |
|---:|---|---|---|---|---|
| 606 | L3 | L3 | L3 | L1+L3, L2+L3 | layer3 dominant |
| 808 | L3 | L3 | L3 | L2+L3 | layer3 dominant |
| 1001 | L3 | L3 | L3 | L1+L3, L2+L3 | layer3 dominant |
| 1879 | L2 | L2 | L2 | L1+L2, L2+L3 | layer2 dominant |

Seeds 606 and 1001 share one exact signature. Seed808 differs in the L1+L3
pair: it reaches only 80.32%, whereas the other two layer-3 models reach
97.22% and 90.87%. Seed1879 has a qualitatively different route centered on
L2. Thus the registered count is three exact signatures, while the main
mechanistic result is two dominant routing classes.

The layer shift is large. In the three historical successful checkpoints,
zeroing or batch-rolling L3 produces 5.37-6.74%, and L3 alone produces
97.17-98.68%. For seed1879, zeroing or batch-rolling L2 produces 12.94% and
11.62%, L2 alone produces 90.92%, L3 alone produces only 13.38%, and removing
L3 still leaves 90.38%.

## Statistical boundary

The formal classification uses the preregistered point-estimate thresholds and
therefore passes exactly as specified. Most decisive effects are far from
their gates. One boundary should nevertheless remain explicit: seed1879's
keep-L2 score is 90.9180%, with a 95% Wilson interval of
[89.5957%, 92.0871%]. Its point estimate passes the 90% sufficiency gate, but
the interval crosses that threshold.

This does not change the registered classification. It limits the precision
of the strongest single-layer-sufficiency wording for seed1879. The L2
necessity and sample-alignment results are not borderline: zero-L2 is 12.94%
([11.55%, 14.46%]) and batch-roll-L2 is 11.62% ([10.30%, 13.08%]). The
initialization-dependent shift in the necessary causal route is therefore
well separated from the 20% gate.

## Scientific conclusion

Level 7.3 confirms the hypothesis registered after Level 7.2:

1. complete persistent Memory causality generalizes across all four examined
   successfully formed checkpoints;
2. the particular layer carrying the necessary and sufficient route does not
   generalize universally across initialization;
3. three checkpoints use a sharply L3-dominant circuit, while seed1879 uses a
   sharply L2-dominant circuit;
4. even within the L3 class, pairwise support structure varies, as shown by
   seed808;
5. high local accuracy under every intervention separates this mechanism from
   ordinary short-context competence.

The correct IST claim is now: **successfully formed models can implement
high-accuracy 16-chunk behavior through causally necessary, sample-specific
persistent Memory, but the layer-level routing of that Memory is
initialization-dependent.** A universal final-layer mechanism is falsified.

This experiment does not repair the formation-reliability failure from Level
7.1. It conditions on four already successful checkpoints, three of which were
historical sources. It therefore establishes mechanistic heterogeneity among
formed models, not the probability that a fresh training run will form one of
these routes. It also makes no comparison with a standard Transformer.

## Registered stop boundary and next study

Level 7.3 is closed under its registered positive classification. Do not
change its thresholds, select a preferred intervention, or modify any source
checkpoint.

The smallest justified follow-up is a separately preregistered precision
replication of seed1879's L2 route on a larger new panel. It should keep the
checkpoint frozen and test intact, complete-Memory controls, zero/roll L2,
keep L2, and matched L3 contrasts. Its purpose would be to narrow the
single-layer-sufficiency interval, not to relabel Level 7.3. A later study can
then track when L2- versus L3-dominant routing emerges in untouched training
initializations.
