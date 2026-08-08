# Level 6.5.4 analysis: independent initialization with hard400

## Result

Five models and probes were independently initialized and trained with the
registered `hard400` protocol under strict mathematical-SDP determinism.

| Seed | Final 2-chunk query | Local | Full pipeline |
| ---: | ---: | ---: | ---: |
| 313 | 7.50% | 100.00% | fail |
| 42 | 3.75% | 100.00% | fail |
| 2026 | 10.00% | 100.00% | fail |
| 7 | 6.25% | 90.00% | fail |
| 1234 | 6.25% | 90.00% | fail |

- 2-chunk formation: 0/5.
- 4-, 8-, and 16-chunk reach: 0/5.
- Maintenance runs: 0.
- Full-pipeline success: 0/5.

## Scaffold diagnostic

During the first 400 probe-supervised steps, neither query nor probe accuracy
rose materially above the 16-class chance level of 6.25%. Maximum warm-up query
accuracy was 7.5--11.25%, and maximum warm-up probe accuracy was 5--11.25%.
After probe loss was removed, query accuracy remained at chance while the local
task reached 90--100%.

Thus the failure occurs before long-context transfer. The hard400 auxiliary
loss did not establish a target-bearing memory representation from random
initialization under this backend and budget. Memory states changed and gates
remained nonzero, but neither the task head nor the probe could decode the
target; state motion alone is not evidence of useful memory.

## Interpretation

The previously observed hard400 success was a single exploratory trajectory
using the memory-efficient attention backward path that PyTorch warned was
non-deterministic. It is evidence that the architecture can enter a useful
basin, but not that hard400 reliably finds that basin. The strict five-seed
experiment rejects hard400 as a reproducible standalone formation method.

This does not invalidate the conditional 2,048-token retention result from
Levels 6.5.1--6.5.3: those experiments showed that an already formed checkpoint
can transfer and remain accurate with a suitable late-stage learning rate. It
does narrow the claim: current evidence demonstrates **capacity and conditional
maintenance**, not reliable formation from random initialization.

## Protocol correction and next experiment

The original successful Level 6.2 checkpoint was not created by hard400 alone.
It inherited the Level 6.1 fixed-marker diagnostic checkpoint before the
random-marker 2/4/8/16 curriculum. Therefore a faithful independent-
initialization confirmation must reproduce the complete Level 6.1 -> Level 6.2
scaffold for every model seed, then use the stabilized `5e-5` 16-chunk stage.

That full curriculum, rather than another hard400 duration sweep, is the next
scientifically justified test.

