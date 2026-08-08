# Level 6.6 post-hoc recovery analysis

## Status

These results are post-hoc diagnostics. They do not alter the registered Level
6.6 full-pipeline result of 2/5.

| Recovery case | Intervention | Result | Final query | Final probe min |
| --- | --- | ---: | ---: | ---: |
| seed7_budget | extend 4-chunk budget | pass | 100.00% | 100.00% |
| seed42_lr | restart 16 chunks at 1e-5 | fail | 92.75% | 93.50% |
| seed2026_withdrawal | slower anneal at 5e-6 | fail | 92.25% | 94.50% |

One of three complete recoveries passed.

## Seed 7: diagnosis confirmed

The source checkpoint already had a final 100% 4-chunk observation but lacked
the second consecutive pass required by protocol. After one additional update,
the recovered evaluation was 98.75% query and 100% probe, satisfying the gate.
The model then passed 8 and 16 chunks in 100 steps each and remained essentially
perfect through withdrawal. Its 400-sample final query and probe accuracy were
both 100%.

This confirms that seed 7 was budget limited rather than structurally unable to
transfer. A transition policy should include more headroom after late recovery,
or use a convergence-based extension rule registered in advance.

## Seed 42: stage diagnosis confirmed, end-to-end rescue failed

Lowering the 16-chunk LR to 1e-5 rescued the curriculum stage in 300 steps. The
model entered withdrawal near 95--98% query. It then became unstable during the
0.1 and zero-probe phases and finished at 92.75% query.

Thus the original 16-chunk degradation was correctly attributed to update
magnitude, but fixing that stage exposed a second withdrawal/maintenance
failure. The case is a partial mechanistic rescue, not a complete pipeline
success.

## Seed 2026: slow withdrawal hypothesis rejected

The slower 0.3/0.2/0.1/0.05/0 schedule at 5e-6 did not recover seed 2026. Final
query improved only from 91.0% to 92.25%. During zero-probe training, individual
80-sample evaluations ranged up to 97.5%, but the independent 400-sample final
evaluation was 92.25%.

This rejects the simple claim that seed 2026 failed only because probe
supervision was removed too quickly. Its representation remains substantially
decodable (94.5% probe minimum), while behavioral readout fluctuates. The
remaining issue is broader late-stage parameter/readout stability.

## Conclusion

The recovery study validates two local diagnoses but only one end-to-end fix:

- transition budget was the full cause for seed 7;
- lower LR fixes seed 42's 16-chunk stage but not later withdrawal;
- slower withdrawal alone does not fix seed 2026.

The scientifically defensible next protocol change is to extend transition
budgets and use the lower 16-chunk LR. Continued per-seed rescue tuning should
stop here. Addressing withdrawal should use a general registered mechanism such
as parameter-group freezing, EMA/SWA weights, or a stability regularizer, then
be rerun across fresh independent initializations.

