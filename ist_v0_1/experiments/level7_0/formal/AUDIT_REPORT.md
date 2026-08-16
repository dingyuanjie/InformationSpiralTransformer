# Level 7.0 evidence audit

## Decision

- Overall: **PASS**.
- Registered claims: 8/8 passed.
- Python syntax checks: 15/15 passed.
- Manifest: 52 files, 1.72 MiB.
- ZIP integrity: **PASS**, 57 members.

## Claim audit

| Claim | Registered status | Checks | Result |
|---|---|---:|---|
| `selective_pollution_defense` | supported | 4/4 | PASS |
| `output_head_rescue` | supported | 5/5 | PASS |
| `task_aligned_context_subspace` | supported | 5/5 | PASS |
| `hard_example_read_access_failure` | supported_boundary | 4/4 | PASS |
| `signed_affine_simplex_obstruction` | supported_boundary | 5/5 | PASS |
| `oracle_not_compiled` | registered_negative | 4/4 | PASS |
| `joint_calibration_coupling_bottleneck` | supported_boundary | 5/5 | PASS |
| `factorized_repair_branch_closed` | registered_negative | 6/6 | PASS |

## Scientific boundary

This audit verifies provenance and consistency of frozen evidence. It does not create a new efficacy result, independently reproduce training, or authorize another router candidate. Registered negative results remain negative. Seed909 and protected tests remain locked.
