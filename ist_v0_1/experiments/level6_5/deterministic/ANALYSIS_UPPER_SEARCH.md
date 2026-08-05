# Level 6.5 deterministic upper scaffold search

## Search result

Seed 313 was tested with hard-stop scaffold durations of 100, 200, 400, 800,
1200, 1600, and 2000 steps, plus 800- and 1600-step linear annealing.

Only `hard400` passed the 2-, 4-, and 8-chunk behavioral gates. Every other
profile stopped at 2 chunks. This is a non-monotonic optimization result: more
direct probe supervision did not monotonically improve memory formation.

## hard400 curriculum

| Chunks | Steps used | Final query | Stage result |
| ---: | ---: | ---: | ---: |
| 2 | 2700 | 95.0% | pass |
| 4 | 1000 | 97.5% | pass |
| 8 | 800 | 97.5% | pass |
| 16 | 1000 | 77.5% | fail |

The 16-chunk stage briefly reached 95% query at step 100, then fell to 72.5--
87.5% for most later evaluations. Local accuracy remained approximately 96--
100%. This is not a failure to reach the long-context solution; it is failure
to preserve that solution under continued 16-chunk optimization.

## Interpretation

The experiment identifies a narrow scaffold region around 400 steps for this
initialization. Shorter schedules were usually too weak, while longer schedules
and gradual annealing did not form the behavioral solution under the same
budget. The result is path dependent rather than a simple supervision-dose
response.

At 16 chunks, the curriculum transfer initially worked and then degraded. The
next controlled variable should therefore be the 16-chunk optimizer update,
not scaffold duration. Reusing the exact `hard400_seed313/stage3.pt` checkpoint
allows learning-rate and update-budget variants to be compared without
repeating stochastic formation.

## Recommended Level 6.5.1

From the shared 8-chunk checkpoint, compare 16-chunk learning rates 1e-4,
5e-5, 2.5e-5, and 1e-5. Record best, final, and post-maintenance query accuracy.
The current 1e-4 trajectory is the control. Select a setting only if it reaches
the gate in consecutive evaluations and remains stable after probe-free
maintenance.

