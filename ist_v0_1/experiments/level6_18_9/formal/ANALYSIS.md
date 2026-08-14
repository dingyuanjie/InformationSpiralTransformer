# Level 6.18.9 formal analysis

## Decision

**Level 6.18.9 completed all 500 optimizer updates but fails the preregistered
stable validation gate.** The terminal message
`stable_task_aligned_read_gate_failed` is an intentional fail-closed decision,
not a Python, CUDA, backward-pass, checkpoint, or file-writing error.

No confirmation panel was opened for the trained model, so the stability streak
remained 0/2. Protected 2,048-example tests and the causal panel were correctly
kept locked. Formal success is false and no protected accuracy should be
inferred from the fixed screen.

The immediate blocker was one discrete 12-chunk screen example. The 128-example
source screen was already 120/128 = 93.75%. The registered candidate threshold
of 0.94 requires at least 121/128 = 94.53% on this panel. No checkpoint changed
the net 12-chunk count above 120/128, so no evaluation satisfied every candidate
condition and the larger confirmation set was never evaluated.

## Execution and boundary audit

Training execution is valid:

- optimizer updates completed: 500/500;
- recorded evaluation rows: 21;
- latest resumable checkpoint written at update 500;
- exactly four tensors changed;
- changed parameters: 16,640;
- every change is under `blocks.2.memory_read.*`;
- original Memory Probe and every frozen parameter remained unchanged;
- persistent Memory was bitwise identical across all 108 audited
  length/chunk/layer rows;
- overall persistent-Memory maximum absolute difference: `0.0`.

| Tensor | Parameters | Maximum absolute change |
|---|---:|---:|
| `memory_read.in_proj_weight` | 12,288 | 0.00853 |
| `memory_read.in_proj_bias` | 192 | 0.00556 |
| `memory_read.out_proj.weight` | 4,096 | 0.00813 |
| `memory_read.out_proj.bias` | 64 | 0.00578 |

The optimizer was therefore active and stayed exactly inside the registered
boundary. The negative formal decision cannot be attributed to a frozen
parameter, missing gradient, illegal mutation, or Memory-state drift.

## Gate calibration context

Before training, the fixed source panels were:

| Panel | Chunks | Samples | Query accuracy | Mean margin |
|---|---:|---:|---:|---:|
| screen | 8 | 128 | 95.31% | 6.4891 |
| screen | 12 | 128 | 93.75% | 5.9770 |
| screen | 16 | 128 | 96.09% | 5.7485 |
| confirmation | 8 | 512 | 97.66% | 6.6710 |
| confirmation | 12 | 512 | 96.09% | 6.2403 |
| confirmation | 16 | 512 | 91.80% | 5.5782 |

The 12-chunk source screen is below the candidate threshold even though the
larger, disjoint source confirmation panel is above 95%. This does not make the
preregistered result invalid; the fail-closed rule was followed exactly. It does
show that the small absolute screen criterion is sensitive to one-example
sampling granularity and is not a clean retention test relative to the source.

## Training and screen trajectory

Across the complete screen trajectory:

| Metric | Minimum | Maximum | Final |
|---|---:|---:|---:|
| 8-chunk query | 95.31% | 95.31% | 95.31% |
| 12-chunk query | 92.97% | 93.75% | 93.75% |
| 16-chunk query | 96.09% | 97.66% | 97.66% |
| 8-chunk margin change | -0.0012 | +0.0991 | +0.0991 |
| 12-chunk margin change | -0.0003 | +0.0860 | +0.0860 |
| 16-chunk margin change | +0.0004 | +0.0996 | +0.0996 |

Every one of the 21 screen rows passed the 8-chunk accuracy condition and the
16-chunk accuracy condition. Nineteen of 21 rows passed the registered
16-chunk margin-improvement screen. Zero rows passed the 12-chunk accuracy
condition, and therefore zero rows passed the complete candidate screen.

At update 500, the fixed screen was:

| Chunks | Source correct | Rescue correct | Source accuracy | Rescue accuracy | Margin change | Cross-entropy change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 122/128 | 122/128 | 95.31% | 95.31% | +0.0991 | +0.00362 |
| 12 | 120/128 | 120/128 | 93.75% | 93.75% | +0.0860 | +0.00096 |
| 16 | 123/128 | 125/128 | 96.09% | 97.66% | +0.0996 | -0.00683 |

The continuous-margin objective moved in the intended direction at all three
lengths and improved two 16-chunk binary decisions on this panel. It did not
produce a net 12-chunk decision change. Cross-entropy improved at 16 chunks but
slightly worsened at 8 and 12 chunks, showing that higher correct-versus-top-
incorrect margin does not imply uniformly better probability calibration.

The recorded small training minibatches remained variable, as expected with
batch size four per length. Evaluation-row training margin ranged from about
4.76 to 6.82, and gradient norm ranged from 0.214 to 1.706. All updates were
finite and completed.

## What was not evaluated

Per the registered fail-closed protocol, the following are absent and must not
be reconstructed from screen data:

- trained-model 512-example confirmation metrics;
- protected 8/12/16 paired tests;
- protected confidence intervals and McNemar tests;
- intact/reset/zero/batch-roll causal panel;
- `task_aligned_read_checkpoint.pt` accepted as a formal rescue.

`read_supervision_latest.pt` is an update-500 diagnostic checkpoint, not a
passed stable checkpoint. No `read_supervision_best.pt` or
`read_supervision_stable.pt` exists because the candidate screen never opened.

## Scientific conclusion

Level 6.18.9 establishes that the Level 6.18.8 task-aligned direction can be
scaled into a steadily larger held-out margin change while preserving exact
persistent Memory and the narrow four-tensor boundary. It also improves the
fixed 16-chunk screen from 96.09% to 97.66%.

It does **not** establish a stable three-length rescue. The 12-chunk discrete
screen never improved above its source count, and no larger trained-model
validation or protected data was opened. The correct conclusion is therefore:

- training executed and learned a measurable screen-level margin effect;
- the formal stable gate failed;
- the update-500 checkpoint remains scientifically interesting but unaccepted;
- more optimizer steps, a different learning rate, or seed909 are not
  authorized by this result.

This is a stricter and more informative result than a runtime failure, but it
must remain a formal negative until the screen-calibration question is resolved
on new data.

## Next experiment

**Level 6.18.9.1 should be a frozen validation-calibration audit of
`read_supervision_latest.pt`, with no further training and no protected-test
access.** It should compare the source and update-500 read checkpoint on at
least two new disjoint, larger validation panels at 8, 12, and 16 chunks.

The audit should preregister:

1. paired 8/12 accuracy non-inferiority relative to source rather than an
   absolute threshold on a 128-example screen;
2. paired 8/12 margin and cross-entropy retention;
3. 16-chunk margin superiority and an absolute 95% accuracy requirement;
4. agreement across independent panels, not only an aggregate average;
5. the existing protected tests and seed909 remain locked.

If the frozen checkpoint fails the larger 16-chunk accuracy or 8/12 retention
criteria, Level 6.18.9 is a genuine rescue failure and optimization should stop.
If it passes on both independent panels, a separately registered gate may then
decide whether protected tests can be opened. No optimizer continuation should
occur before this audit.
