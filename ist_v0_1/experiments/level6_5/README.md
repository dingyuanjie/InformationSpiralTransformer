# Level 6.5: minimum memory scaffold

Level 6.4 found a 5/5 versus 0/5 separation: established memory survived after
probe-loss removal, while memory did not reliably form from scratch. Level 6.5
measures how little direct memory supervision is needed to prevent early memory
path suppression.

## Search profiles

| Profile | Stage-1 probe schedule | Weighted probe steps |
| --- | --- | ---: |
| `zero` | no probe loss | 0 |
| `hard50` | weight 0.5 for 50 steps, then 0 | 25 |
| `hard100` | weight 0.5 for 100 steps, then 0 | 50 |
| `hard200` | weight 0.5 for 200 steps, then 0 | 100 |
| `hard400` | weight 0.5 for 400 steps, then 0 | 200 |
| `hard800` | weight 0.5 for 800 steps, then 0 | 400 |
| `anneal200` | linearly anneal 0.5 to 0 over 200 steps | 50 |

`hard100` and `anneal200` have approximately equal integrated supervision, so
they compare abrupt removal with gradual withdrawal.

Every run starts from an independent random model initialization. A curriculum
stage passes only after probe loss is zero and query accuracy is at least 95%
in two consecutive evaluations. The target appears only in chunk 1 and the
query only in the final chunk, so this is the direct behavioral test of
persistent memory. Minimum per-chunk linear-probe accuracy is retained as a
diagnostic, not a gate: a probe trained for only a short warm-up and then frozen
can underfit a useful memory representation. Runs that pass 2/4/8/16 chunks
receive another 500 zero-probe maintenance steps and a 50-batch final
evaluation.

The original pilot under `formal/` used probe accuracy as an additional gate.
It found stable 100% query accuracy for `hard50` but incorrectly stopped the
curriculum because its frozen probe remained near chance. Corrected runs are
written separately under `formal_query_gate/` to preserve the audit trail.

## Phase A: one-seed threshold search

```powershell
python run_level6_5_local.py
```

## Phase B: confirm the smallest successful profiles

After Phase A, select the boundary profile and its nearest failed/successful
neighbors. Example:

```powershell
python run_level6_5_local.py --profiles hard100 anneal200 hard200 --seeds 313 42 2026 7 1234
```

Existing completed profile/seed folders are reused unless `--force` is passed.
The repeated corrected `hard50` run reached only 67.5% query at step 3000,
despite using the same nominal seed as the original 100% run. The common seed
helper fixed RNG state but did not enable deterministic CUDA kernels. Level 6.5
therefore fixes RNG state and requests deterministic PyTorch/CUDA behavior by
default. A later smoke test showed that `warn_only=True` still allowed the
non-deterministic memory-efficient attention backward kernel. The five-seed
results remain valid as multi-initialization observations, but they are not
bitwise-repeatability evidence. Current scripts force the mathematical SDP
backend, use strict deterministic algorithms, and disable TF32. Use
`--allow-nondeterministic` only for an explicitly exploratory run.

Deterministic results are written under `experiments/level6_5/deterministic/`.
