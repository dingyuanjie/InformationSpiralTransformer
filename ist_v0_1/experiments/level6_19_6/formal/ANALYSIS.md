# Level 6.19.6 formal analysis

## Decision

- Integrity: **PASS**.
- Factorized signed recovery: **FAIL**, 24.4056% versus the registered 25%
  minimum (shortfall 0.5944 percentage points).
- Six-way specificity: **FAIL**, four of six Holm-corrected contrasts passed.
- Full-accuracy noninferiority: **PASS**, +0.1221 percentage points relative to
  source, with a 95% bootstrap CI of [0.0000, +0.2686] percentage points.
- Registered conjunction: **FAIL**.
- Registered classification: `factorized_repair_failed_stop_branch`.
- Registered next boundary: stop this router-repair branch; do not introduce a
  second composition formula and do not open seed909.

This is a valid scientific failure rather than a failed or incomplete run. All
integrity checks passed before the registered efficacy gate was applied.

## Integrity

The seed707 trunk, existing probes, Level 6.19.4 parent routers, and Level
6.19.5 final probes retained their frozen fingerprints. All frozen parameters
remained `requires_grad=False`, and parent-router fingerprints were unchanged.
The diagnostic split used the fresh registered seed `6196100` with 4,096
examples.

No training, calibration, threshold search, candidate selection, optimizer
search, or model search occurred. Exactly one repair candidate was evaluated.
At inference it used no target label, rival class, correctness flag, Oracle
dose, or Oracle direction. The frozen classifier was not used as a gate. The
failed Level 6.18.9 candidate was excluded; seed909 and protected tests remained
locked.

Numerical reconstruction audits also passed:

- full signed Oracle maximum L2 reconstruction error: `3.5763e-6`;
- signed and residual unit-direction maximum errors: `1.1921e-7` each;
- candidate, residual, shuffled, rolled, and head-permuted dose maximum errors:
  `2.3842e-7` each;
- source replay maximum absolute difference: `0.0625`;
- aggregate integrity result: **PASS**.

## Fresh diagnostic panel

The fresh panel contained 293 source errors at 92.8467% source accuracy. Of
these, 232 were Memory-decodable and formed the registered primary population.
`Primary fixed` is the fraction of those 232 cases corrected. Margin gain and
Oracle recovery are measured on that same primary population.

| Condition | Full accuracy | Primary fixed | Primary margin gain | Oracle recovery | Corrections | Regressions | Context L2 full / primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 92.8467% | 0.00% | 0.000000 | 0.00% | 0 | 0 | 0.0000 / 0.0000 |
| Frozen signed router | 92.8955% | 1.29% | 0.036682 | 22.26% | 3 | 1 | 0.5922 / 0.7427 |
| Factorized signed candidate | **92.9688%** | **2.59%** | **0.040224** | **24.41%** | **6** | **1** | 0.9298 / 0.8317 |
| Factorized residual control | 92.9443% | 2.59% | 0.035308 | 21.42% | 6 | 2 | 0.9298 / 0.8317 |
| Shuffled Memory | 92.8711% | 0.43% | 0.011984 | 7.27% | 1 | 0 | 0.9081 / 0.8002 |
| Rolled delta | 92.7734% | 0.00% | -0.002455 | -1.49% | 0 | 3 | 0.9298 / 0.9249 |
| Head-permuted | 92.8467% | 1.29% | 0.009921 | 6.02% | 3 | 3 | 0.9298 / 0.8317 |
| Full signed Oracle | 93.2617% | 7.33% | 0.164815 | 100.00% | 17 | 0 | 0.9511 / 0.8727 |

The candidate made six corrections and one regression, a net gain of five
correct predictions. Its 24.4056% margin recovery nevertheless remained below
the hard 25% gate; the near miss is not rounded up or treated as a pass.

## Registered specificity

All estimates are candidate-minus-control paired deployed-margin contrasts on
the 232-case primary population. The registered rule required all six effects
to be positive and significant after Holm correction.

| Contrast | Estimate | 95% CI | Raw sign-flip p | Holm p | Result |
|---|---:|---:|---:|---:|---|
| Candidate vs source | +0.040224 | [0.032576, 0.048189] | 0.000100 | 0.000600 | PASS |
| Candidate vs frozen signed router | +0.003542 | [-0.003147, 0.009305] | 0.290171 | 0.290171 | **FAIL** |
| Candidate vs factorized residual | +0.004916 | [-0.001052, 0.010955] | 0.124388 | 0.248775 | **FAIL** |
| Candidate vs shuffled Memory | +0.028240 | [0.021042, 0.035508] | 0.000100 | 0.000600 | PASS |
| Candidate vs rolled delta | +0.042679 | [0.033132, 0.052768] | 0.000100 | 0.000600 | PASS |
| Candidate vs head-permuted | +0.030303 | [0.023694, 0.037223] | 0.000100 | 0.000600 | PASS |

The candidate therefore has a real Memory-, alignment-, and head-structure-
dependent effect: shuffling Memory, rolling the delta, or permuting heads
removes a significant part of the gain. However, its small numerical gains over
the prior frozen signed router and the residual-direction control are not
statistically established. The registered six-way specificity gate fails.

## Full-accuracy safety

Full accuracy rose from 92.8467% to 92.9688%, an estimate of +0.1221 percentage
points. The registered noninferiority floor was -0.25 percentage points, so the
95% bootstrap CI of [0.0000, +0.2686] percentage points passes noninferiority.
There were seven discordant decisions: six source errors corrected and one
source-correct example regressed (two-sided exact McNemar p=0.125). This is a
safety pass, not evidence that the complete efficacy conjunction passed.

## Frozen classifier audit

The frozen error-state classifier remained strongly enriched on this new
split:

- prevalence: 5.6641% (232/4,096);
- AUROC: 0.9636;
- average precision: 0.6799;
- precision among the top 232 predictions: 0.6466;
- fixed-prevalence lift: 11.415x.

These values support the earlier conclusion that the relevant error state is
observable. The classifier output did **not** gate, scale, or select the repair
candidate, so no post-hoc classifier threshold may be introduced from these
results.

## Scientific conclusion

Level 6.19.5 showed that error state, dose, and a supervised signed direction
are individually observable. Level 6.19.6 now shows that simply multiplying the
two frozen factor outputs does not compile them into a successful label-free
read. The factorized candidate preserves global accuracy and exhibits genuine
structured Memory dependence, but it neither reaches the registered Oracle-
recovery threshold nor establishes improvement over both the previous router
and the residual basis.

The remaining bottleneck is therefore not the absence of a decodable Memory
signal. It is the deployment-time coupling/calibration of dose with the correct
signed direction. Strong component probes and classifier enrichment are not
sufficient evidence that their frozen composition is an effective causal
router.

The result must not be reclassified as a near-pass. The old router, residual
control, classifier gate, dose cap, or another composition formula cannot be
selected after viewing this split.

## Next experiment

There is no next experiment inside this router-repair branch. Per the registered
failure rule, the branch is closed: do not introduce a second repair formula,
do not tune a classifier gate or threshold, and do not open seed909. The next
project-level action should be result consolidation and reproducibility
packaging; any future architectural hypothesis must begin as a separately
pre-registered research phase and cannot reuse this formal split for selection.
