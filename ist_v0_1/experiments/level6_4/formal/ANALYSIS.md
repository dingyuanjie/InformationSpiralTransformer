# Level 6.4 final analysis: maintenance versus spontaneous formation

## Protocol

- Five seeds: 313, 42, 2026, 7, 1234
- Chunk size: 128
- Maintenance: load the passed Level 6.2 16-chunk checkpoint, freeze the
  diagnostic probe, and train for 500 steps with zero probe loss.
- Scratch-zero: independent random initialization, frozen random probe, zero
  probe loss for the entire 2/4/8/16-chunk curriculum.
- Pass criteria for maintenance: final query >= 95% and minimum probe >= 90%.
- Scratch stages require two consecutive evaluations with query >= 95%.

## Main result

| Mode | Passed | Success rate |
| --- | ---: | ---: |
| Maintenance | 5/5 | 100% |
| Scratch-zero | 0/5 | 0% |

All maintenance runs remained strong for 500 zero-probe steps. Final 16-chunk
query accuracy was 98.5--99.75%, and minimum probe accuracy was 92.25--98.75%.
Even the worst intermediate maintenance evaluation retained 96.25% query and
91.25% probe accuracy.

All scratch-zero runs failed the first 2-chunk curriculum gate after 3,000
steps. Four seeds stayed near the 16-class chance level (6.25%) on the query
task despite learning the local task. Their final query accuracies were 2.5--
7.5%, while local accuracy was 86.25--95%.

Seed 2026 is a meaningful near-success: query accuracy rose to 81.25% at the
end of training, but it did not reach the 95% stability gate and its frozen
probe remained at 5%. This suggests a delayed, partially useful solution may
occasionally begin to form without scaffolding. It does not make spontaneous
formation reliable under the tested budget.

## Interpretation

Level 6.4 cleanly separates two claims:

1. **Maintenance is robust.** Once probe supervision and curriculum establish
   a persistent-memory solution, query/local training can preserve it across
   new data streams without continued probe loss.
2. **Formation is not robust.** Query and local losses alone did not reliably
   create that solution from random initialization. The observed success rate
   was 0/5 under the registered gate and training budget.

In most scratch runs, memory gate means fell substantially from their roughly
0.5 initialization, while the local task improved. This is consistent with an
optimization race: the local pathway becomes useful first, and the model then
suppresses the initially noisy memory pathway. A changing memory state or a low
first/final memory cosine is not evidence of useful storage when query and probe
accuracy remain near chance.

The evidence therefore supports direct memory supervision as a **formation
scaffold**, not as a permanently required auxiliary objective.

## Limits

- The five maintenance runs share one originally trained checkpoint, so they
  test robustness across subsequent data streams, not across independently
  scaffolded model initializations.
- The task is synthetic and uses one architecture scale.
- A 0/5 scratch result rejects reliable formation under this protocol but does
  not prove that spontaneous formation has zero probability.
- The frozen random probe is not a definitive representation test for scratch
  runs; query accuracy is the primary formation metric there.

## Recommended Level 6.5

Measure the minimum scaffold needed to prevent early memory-path suppression.
Use several probe-loss warm-up durations or annealing schedules, then remove
probe loss completely and apply the same 16-chunk maintenance test. Include a
zero-scaffold control and multiple initialization seeds. The primary outcomes
should be formation success rate, supervision cost, final zero-probe query
accuracy, and memory-gate trajectories.

