# Corrected hard50 repeat

The corrected query-gated repeat did not reproduce the original pilot's 100%
query result. It ended the 2-chunk stage at 67.5% query and therefore did not
enter the longer curriculum.

This repeat was not a flat chance-level failure. Its final three evaluations
rose from 15% to 45% to 67.5%, consistent with delayed formation that had not
crossed the registered 95% gate before the 3,000-step budget ended.

The two nominally identical seed-313 runs diverged because RNG seeds were fixed
but deterministic CUDA algorithms were not enabled. The observations should be
reported as exploratory outcomes: one formed a stable behavioral solution and
one began a late transition. They do not establish a reproducible 50-step
threshold.

Subsequent confirmation runs enable deterministic algorithms, deterministic
cuDNN, a fixed cuBLAS workspace, and disabled TF32. They are stored separately
under `../deterministic/`.
