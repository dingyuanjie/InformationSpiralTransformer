# Level 6.6 analysis: full independent formation validation

## Registered result

Five independently initialized models followed the complete fixed-marker,
random-marker curriculum, 16-chunk stabilization, probe withdrawal, and
maintenance protocol.

| Outcome | Count |
| --- | ---: |
| Fixed-marker Level 6.1 passed | 5/5 |
| Random-marker 2-chunk passed | 5/5 |
| 4-chunk passed | 4/5 |
| 8-chunk passed | 4/5 |
| 16-chunk passed | 3/5 |
| Final withdrawal/maintenance passed | 2/5 |
| Full registered pipeline | **2/5 (40%)** |

The two complete successes were seed 313 and seed 1234. Their final 400-sample
query accuracies were 99.5% and 95.0%; final minimum probe accuracies were 99.0%
and 95.75%.

## What the fixed-marker scaffold changed

The hard400-only experiment formed useful 2-chunk memory in 0/5 independent
initializations. With the faithful Level 6.1 fixed-marker stage, all five models
reached essentially 100% fixed-marker query and probe accuracy, and all five
subsequently passed random-marker 2-chunk formation.

This is the strongest formation result in Level 6: task simplification before
random-marker training converts formation from 0/5 to 5/5 at the first random
stage. Direct probe duration alone was not an adequate substitute.

## Failure modes

### Seed 7: budget-limited 4-chunk transition

Seed 7 ended the 4-chunk budget with query, local, and probe all at 100%. It was
marked failed because the preregistered gate requires two consecutive passing
evaluations and only the final evaluation crossed the threshold. Its trajectory
recovered from near chance at steps 100--400 to 90% at step 900 and 100% at step
1000. This is a strict protocol failure, but the evidence suggests insufficient
transition budget rather than lack of capacity.

### Seed 42: genuine 16-chunk optimization degradation

Seed 42 entered 16 chunks at 96.25% query but declined under continued updates,
ending at 82.5%. Probe accuracy remained 90--96%, so the memory representation
was partly retained while task readout/optimization degraded. This reproduces
the late-curriculum stability problem on an independent model.

### Seed 2026: withdrawal instability

Seed 2026 passed all curriculum stages, including 97.5% at 16 chunks, but ended
probe-free maintenance at 91.0% query and 92.25% minimum probe accuracy. It
retained substantial memory but failed the 95% final gate after withdrawal.

## Interpretation

The experiment establishes three distinct facts:

1. The architecture can form cross-chunk memory reliably when given the full
   fixed-marker scaffold: 5/5 random-marker two-chunk formation.
2. The complete 2,048-token, probe-free pipeline is possible across independent
   initializations, not just one checkpoint: 2/5 strict successes.
3. The current training protocol is not yet robust: longer-context transition
   budgets, late-stage optimization, and supervision withdrawal each account
   for separate failures.

The appropriate claim is therefore **demonstrated independent reproducibility
with 40% end-to-end success**, not universal stability.

## Recommended next experiment

Preserve this 2/5 result as the primary confirmation. A separately labelled
recovery study can test the three diagnosed interventions without changing the
registered outcome:

- extend the 4-chunk budget for seed 7;
- restart seed 42's stage-3 checkpoint with a lower 16-chunk LR (`1e-5`);
- use a slower withdrawal schedule and/or lower withdrawal LR for seed 2026.

These are targeted post-hoc controls. If they rescue their respective failures,
their policies must then be preregistered and rerun across all five fresh seeds
before claiming a higher pipeline success rate.

