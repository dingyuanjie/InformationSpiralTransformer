# Level 6.18.9.1 formal analysis

## Decision

**Level 6.18.9.1 fails the frozen validation-calibration audit.** The registered
classification is `frozen_validation_calibration_failed`. Both independent
panels pass all 8- and 12-chunk non-inferiority checks, and both show strong
continuous-margin improvement at 16 chunks. However, the candidate reaches only
92.29% and 92.04% 16-chunk accuracy, far below the required 95% on both panels.

The existing protected tests were not opened, seed909 remained locked, and no
parameter was updated in this audit. The registered next boundary is to reject
the Level 6.18.9 checkpoint as a stable rescue and stop Memory-read optimizer
work.

## Checkpoint and Memory integrity

All integrity checks passed:

- candidate checkpoint is Level 6.18.9 update 500;
- exactly four final `memory_read` tensors differ from source;
- changed parameters: 16,640;
- original Memory Probe and every other parameter are unchanged;
- persistent Memory is bitwise identical across all 108 audited
  length/chunk/layer rows;
- maximum persistent-Memory difference: `0.0`;
- no forward, backward, optimizer, or checkpoint error occurred.

| Tensor | Parameters | Maximum absolute change |
|---|---:|---:|
| `memory_read.in_proj_weight` | 12,288 | 0.00853 |
| `memory_read.in_proj_bias` | 192 | 0.00556 |
| `memory_read.out_proj.weight` | 4,096 | 0.00813 |
| `memory_read.out_proj.bias` | 64 | 0.00578 |

The formal failure is behavioral, not an implementation or boundary failure.

## Panel A

Panel A contains 2,048 new examples at each length.

| Chunks | Source accuracy | Candidate accuracy | Accuracy change, 95% CI | Margin change, 95% CI | Cross-entropy change, 95% CI | Pass |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 97.75% | 97.80% | +0.049 pp [0.000, +0.146] | +0.0849 [+0.0775, +0.0924] | -0.00014 [-0.00139, +0.00094] | yes |
| 12 | 96.48% | 96.53% | +0.049 pp [0.000, +0.146] | +0.0875 [+0.0798, +0.0954] | +0.00042 [-0.00138, +0.00268] | yes |
| 16 | 92.24% | 92.29% | +0.049 pp [-0.098, +0.244] | +0.0798 [+0.0718, +0.0880] | -0.00206 [-0.00432, +0.00024] | **no** |

All paired non-inferiority and 16-chunk margin-superiority checks pass. The
16-chunk margin sign-flip test has `p=0.00010`. The sole failed check is absolute
candidate accuracy: 1,890/2,048 = 92.29%, whereas 95% requires at least
1,946/2,048. The candidate is short by 56 correct examples.

At 16 chunks there are two source-wrong/candidate-correct cases and one reverse
case, yielding only one net correction. Exact McNemar `p=1.0`.

## Panel B

Panel B contains another 2,048 examples per length, with disjoint seeds.

| Chunks | Source accuracy | Candidate accuracy | Accuracy change, 95% CI | Margin change, 95% CI | Cross-entropy change, 95% CI | Pass |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 97.02% | 97.02% | 0.000 pp [0.000, 0.000] | +0.0871 [+0.0796, +0.0943] | +0.00171 [+0.00023, +0.00344] | yes |
| 12 | 96.58% | 96.53% | -0.049 pp [-0.244, +0.098] | +0.0907 [+0.0822, +0.0994] | +0.00020 [-0.00301, +0.00324] | yes |
| 16 | 92.04% | 92.04% | 0.000 pp [-0.146, +0.146] | +0.0919 [+0.0831, +0.1015] | -0.00166 [-0.00338, +0.00048] | **no** |

Again, all non-inferiority and continuous-margin checks pass. The 16-chunk
margin sign-flip test has `p=0.00010`. The sole failed check is absolute
accuracy: 1,885/2,048 = 92.04%, 61 correct examples short of the required
1,946.

At 16 chunks there is one correction and one regression, so the net accuracy
change is exactly zero. Exact McNemar `p=1.0`.

## Cross-panel agreement

The independent panels agree on every substantive conclusion:

1. 8-chunk accuracy is retained within the registered 1-point margin.
2. 12-chunk accuracy is retained; the earlier 120/128 screen failure was indeed
   a small-panel absolute-threshold artifact rather than a large 12-chunk
   regression.
3. Margin increases by approximately 0.08-0.09 at every length.
4. Cross-entropy remains within the registered +0.01 non-inferiority margin.
5. 16-chunk accuracy remains near 92%, with essentially no net improvement.
6. Neither panel is close to the required 95% absolute 16-chunk threshold.

The favorable 128-example Level 6.18.9 screen, where 16-chunk accuracy reached
125/128 = 97.66%, did not replicate on either larger panel. That screen contained
only five source errors and was not representative of the roughly 8% error rate
seen in both 2,048-example calibration panels.

The result cannot be rescued by pooling the panels: the protocol requires
independent replication, and each panel fails the same absolute condition.

## Why margin improved without correcting errors

A post-hoc descriptive stratification explains the objective-behavior split.
It is not an additional confirmatory test.

At 16 chunks:

| Panel | Mean margin gain on source-correct examples | Mean gain on source-wrong examples | Net corrected examples |
|---|---:|---:|---:|
| A | +0.0853 | +0.0154 | +1 |
| B | +0.0996 | +0.0030 | 0 |

Most of the mean-margin increase reinforces already confident, already correct
decisions. In Panel A, 71.0% of source-correct examples have positive margin
change, compared with 58.5% of source-wrong examples; the gains on wrong
examples are much smaller. In Panel B, 71.5% of correct examples improve, while
wrong examples average almost zero change.

The same pattern is visible across source-margin strata: high-margin examples
receive much larger mean changes than the lowest-margin examples. The registered
mean-margin objective therefore scales the task-aligned direction confirmed in
Level 6.18.8, but it does not concentrate sufficient update on hard or incorrect
queries. A significant population mean is not equivalent to a decision-boundary
rescue.

## Relation to Levels 6.18.8 and 6.18.9

Level 6.18.8 correctly established a real task-aligned component in the read
context. Level 6.18.9 then amplified held-out margin while staying inside the
four-tensor route boundary. Level 6.18.9.1 does not negate either mechanism
result. It shows the limit of that mechanism as a rescue:

- the direction is real;
- its mean continuous effect replicates;
- 8/12 retention is acceptable;
- the learned change mostly strengthens existing correct predictions;
- it produces almost no new 16-chunk decisions on large independent panels;
- it cannot close the approximately three-point gap to the 95% criterion.

Thus the update learned a calibration/margin effect rather than a stable error-
correction mechanism.

## Scientific conclusion

The Level 6.18.9 checkpoint is rejected as a stable rescue. Formal protected
tests remain unopened, and there is no accepted `task_aligned_read_checkpoint`
for later initialization or seed transfer.

The narrow final `memory_read` boundary has now received both ordinary query
supervision (Level 6.18.5) and task-aligned margin/Memory-contrast supervision
(Level 6.18.9). Neither produced stable 16-chunk recovery. Continuing optimizer
steps, changing learning rate, retuning loss weights, or reopening another read-
only search would be post-hoc optimization and is not supported.

The valid positive result that remains is mechanistic: persistent Memory holds
the answer and a small task-aligned read direction exists. The negative result
is equally important: the final read module alone is not sufficient to convert
that information into robust long-context decisions.

## Next boundary

Per the preregistered decision, **stop final Memory-read optimization and keep
seed909 and the protected tests locked**.

The next scientific stage should not be Level 6.18.9.2 optimizer tuning. A clean
next milestone is **Level 6.19: hard-example/residual-path diagnosis**, entirely
frozen at first. It should compare source-correct and source-wrong 16-chunk
examples across:

1. persistent-Memory decodability and slot composition;
2. read attention/context and query-hidden margins;
3. gradient accessibility of the correct Memory code;
4. residual-stream and FFN/norm interactions;
5. matched controls stratified by source confidence.

Only if that frozen study localizes a reproducible downstream obstruction
beyond `memory_read` should a broader architectural boundary be preregistered.
The Level 6.18 read-rescue branch is complete.
