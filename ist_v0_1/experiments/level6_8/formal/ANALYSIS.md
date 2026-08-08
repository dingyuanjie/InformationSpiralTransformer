# Level 6.8 analysis: behavior-first unified protocol

## Registered result

Five fresh independent model seeds used the Level 6.7 budgets, an 8-chunk LR
of `5e-5`, a 16-chunk LR of `1e-5`, no EMA, and behavioral query accuracy as
the primary gate.

| Outcome | Count |
| --- | ---: |
| Fixed-marker formation | 5/5 |
| Random-marker 2 chunks | 5/5 |
| 4 chunks | 5/5 |
| 8 chunks | 4/5 |
| 16 chunks | 3/5 |
| Final behavioral success | **3/5 (60%)** |
| Final probe diagnostic success | 1/5 |

The three models that completed withdrawal all passed the primary final gate:

| Seed | Final query | Final probe min | Final local |
| ---: | ---: | ---: | ---: |
| 606 | 97.75% | 98.00% | 99.00% |
| 808 | 96.25% | 14.50% | 99.75% |
| 1001 | 97.25% | 24.75% | 99.75% |

Mean final query among completed models was 97.08%; the worst was 96.25%.

## Probe decoupling

Seeds 808 and 1001 solve the 2,048-token behavioral task at 96--97% while the
mean-pooled linear memory probe is only 14.5--24.75%. Since chunks communicate
only through the persistent memory argument, the target must still traverse the
memory mechanism. The result shows that the current probe representation
(mean-pooling slots from all layers followed by one linear map) does not track
all useful memory codes. Information may be slot-specific, distributed, or
decoded nonlinearly by the model.

This validates the behavior-first reporting decision. Probe accuracy remains a
useful diagnostic when high, but low probe accuracy cannot be interpreted as
absence of functional memory.

## Failure modes

### Seed 707: stable 8-chunk underfitting

Seed 707 remained mostly between 75% and 92.5% query throughout 1,500 8-chunk
steps and ended at 85%. Local accuracy stayed high. Lowering the LR eliminated
the violent collapse seen for Level 6.7 seed 303 but did not find a passing
solution within budget. This is a plateau/optimization-basin failure.

### Seed 909: late 16-chunk crossing without confirmation

Seed 909 reached 95% query at step 1400 and ended at 90%. It failed the two-
consecutive-evaluation rule. This resembles a late, noisy transition rather
than a hard capacity limit.

## Comparison with Level 6.7

Level 6.7 achieved 3/5 strict successes and 4/5 behavioral successes; Level 6.8
achieved 3/5 behavioral successes on a different fresh seed set. Because both
the seeds and primary gate changed, this is not a paired superiority test.
The evidence does not show that lowering the 8-chunk LR improves overall
success probability, only that it changes the failure mode from catastrophic
instability to underfitting in the observed failed seed.

Across both fresh-seed experiments, 7 of 10 models achieved behavioral
end-to-end success under closely related protocols. This pooled number is
descriptive, not a preregistered combined estimate.

## Conclusion and next step

The current evidence supports reliable initial formation (10/10 fixed-marker
and 2-chunk success across Levels 6.7--6.8) and substantial but imperfect
2,048-token end-to-end reproducibility. Further per-stage LR tuning risks
overfitting the synthetic benchmark.

The next experiment should be causal rather than another optimizer sweep. On
the successful Level 6.8 checkpoints, evaluate intact memory against reset,
zeroed, and batch-shuffled memory between chunks. A causal memory mechanism
should retain high query accuracy only when memory identity and continuity are
preserved, while local accuracy remains high in all conditions. This will
directly establish that long-range behavior is carried by persistent memory and
clarify the apparent probe discrepancy.

