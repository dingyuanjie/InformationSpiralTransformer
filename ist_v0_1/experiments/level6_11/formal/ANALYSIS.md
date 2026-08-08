# Level 6.11 analysis: selective causal memory intervention

## Layer-level result

All three models pass the preregistered final-layer localization criterion.

| Seed | Intact | Zero layer 0 | Zero layer 1 | Zero layer 2 | Only layer 0 | Only layer 1 | Only layer 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 606 | 96.25% | 96.50% | 97.75% | 4.25% | 4.00% | 4.25% | 98.25% |
| 808 | 96.50% | 97.25% | 81.75% | 4.50% | 6.00% | 6.50% | 98.50% |
| 1001 | 97.75% | 97.50% | 91.75% | 7.75% | 7.50% | 7.50% | 97.75% |

Zeroing final-layer memory collapses query to the 6.25% chance level, while
preserving only final-layer memory retains or slightly improves performance.
Layers 0 and 1 cannot drive the task alone. This converts Level 6.10's
decodability localization into a causal localization result.

Layer-1 removal has a model-dependent effect when layer 0 remains present
(notably seed 808), even though layer 2 alone is sufficient. These
counterfactual combinations are off the training distribution, so the safest
claim is that final-layer memory is necessary and sufficient under the tested
interventions; lower memories can modulate the read path in some models.

## Slot-count result

Small final-layer subsets retain substantial behavior, but the precise slots
matter.

| Seed | Minimum fixed prefix >=90% | Minimum random K with all 3 subsets >=90% |
| ---: | ---: | ---: |
| 606 | 2 | 8 |
| 808 | 4 | 8 |
| 1001 | 4 | 2 |

Eight randomly selected final-layer slots are sufficient across every tested
model and random subset. In favorable models/subsets, one or two slots are
enough.

## Single-slot causal heterogeneity

Single-slot examples show large variation:

- seed 606: tested individual slots 0, 5, 14, and 28 yield 88.75--91.75%;
- seed 808: slots 25 and 29 yield 75.25--77%, while slots 0 and 14 are at chance;
- seed 1001: slots 2 and 19 yield 98.25--99%, while slots 0 and 18 are at chance.

This is not contradictory to Level 6.10. Tomography showed that nearly every
final-layer slot contains linearly decodable target information. Level 6.11
shows that the model's own learned read path does not use every decodable slot
equally. Information availability and causal utilization are distinct.

The original frozen mean probe remains low after severe slot masking and should
not be interpreted here; masking changes its input distribution, and Level 6.10
already established its alignment problem.

## Non-monotonic subset effects

Accuracy is not perfectly monotonic with the number of retained slots. For
example, some four-slot subsets outperform some eight-slot subsets. Possible
causes include learned slot-specific read weights, constructive/destructive
interactions, attention normalization, and the fact that zero-masked states are
off-distribution. The robust conclusion is a sufficiency bound (eight random
slots), not a simple claim that every added slot helps.

## Combined Level 6.9--6.11 conclusion

The evidence now supports a detailed causal picture:

1. sample-specific persistent memory is necessary for cross-chunk behavior;
2. final-layer memory is necessary and sufficient;
3. target information is decodable from nearly every final-layer slot;
4. the behavioral read path causally favors particular slots and slot
   combinations;
5. eight random final-layer slots preserve >=90% query across all tested
   models/subsets, indicating substantial redundancy despite causal
   heterogeneity.

## Recommended next experiment

Run an exhaustive final-layer causal slot map: evaluate all 32 keep-one and all
32 leave-one-out interventions for each model, then correlate causal effects
with memory-read attention or fusion weights. This will identify whether the
model uses a sparse set of privileged slots, distributed combinations, or
model-specific routing patterns.

