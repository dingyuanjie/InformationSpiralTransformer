# Level 6.7 analysis: unified robust protocol

## Registered result

Five fresh independent model seeds were trained with extended transition
budgets, lower 16-chunk and withdrawal learning rates, and a preregistered EMA
final model.

| Outcome | Count |
| --- | ---: |
| Fixed-marker formation | 5/5 |
| Random-marker 2 chunks | 5/5 |
| 4 chunks | 5/5 |
| 8 chunks | 4/5 |
| 16 chunks | 4/5 |
| Strict raw final success | 3/5 |
| Strict EMA final success | **3/5 (60%)** |

The strict gate requires both query >=95% and minimum probe accuracy >=90%.
Level 6.7 improves the registered end-to-end result from Level 6.6's 2/5 to
3/5 on a fresh seed set.

## Final models

| Seed | Raw query | Raw probe min | EMA query | EMA probe min | Strict result |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 98.50% | 88.00% | 97.50% | 89.25% | fail |
| 202 | 97.50% | 98.25% | 97.50% | 98.00% | pass |
| 303 | n/a | n/a | n/a | n/a | 8-chunk failure |
| 404 | 99.50% | 99.75% | 99.50% | 99.75% | pass |
| 505 | 98.50% | 99.00% | 99.00% | 99.25% | pass |

Among the four models that completed the curriculum, mean raw query was 98.5%
and mean EMA query was 98.375%. EMA did not increase the success count and did
not improve mean query accuracy.

## Behavioral versus probe result

Seed 101 is a registered strict failure solely because EMA probe minimum was
89.25%, just below the 90% auxiliary gate. Its behavioral query accuracy was
97.5% (98.5% for raw weights). Therefore:

- preregistered strict full-pipeline success is 3/5;
- behavioral query success is 4/5.

Both numbers should be reported. The strict result must not be rewritten, while
the behavioral result matters because the query is the direct cross-chunk task
and the linear probe is an auxiliary representation diagnostic.

## Seed 303 failure

Seed 303 entered 8 chunks with 93.75% query and reached isolated 96.25%
evaluations at steps 100 and 1000, but never produced two consecutive passes.
Its trajectory repeatedly collapsed and recovered, then ended at 25% query and
30% probe accuracy. Local accuracy also fell to 37.5% at the final evaluation.

This is a genuine 8-chunk optimization instability, not insufficient memory
capacity or withdrawal failure. The 8-chunk LR remained 2.5e-4, substantially
higher than the stabilized 16-chunk LR of 1e-5. Level 6.7 moved the late-stage
bottleneck one curriculum level earlier.

## Conclusions

1. The unified protocol improves strict independent end-to-end reproducibility
   from 40% to 60%, and behavioral reproducibility to 80%.
2. Extended budgets solve late-transition cases such as the Level 6.6 seed-7
   failure.
3. EMA at decay 0.995 provides no clear benefit in this experiment.
4. The remaining structural training problem is the 8-chunk update magnitude.
5. Probe accuracy should remain a reported diagnostic, but future primary gates
   should be declared behavior-first before running, consistent with the
   corrected Level 6.5 reasoning.

## Recommended next protocol

Use fresh seeds and retain the Level 6.7 budgets, but lower the 8-chunk LR from
2.5e-4 to 5e-5 (with 16 chunks at 1e-5). Remove EMA unless used as an explicit
ablation. Preregister query >=95% as the primary end-to-end gate and probe
minimum >=90% as a secondary diagnostic. This directly targets the only genuine
curriculum failure observed here.

