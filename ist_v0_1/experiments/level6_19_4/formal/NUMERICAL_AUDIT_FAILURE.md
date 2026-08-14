# Level 6.19.4 first formal run: retained numerical-audit failure

This directory is not the canonical scientific result. It is retained so the
recovery remains auditable.

The original formal run completed every registered stage with 2,048
accelerated projected-gradient iterations. Its convex simplex solver narrowly
missed both hard numerical gates on the 241 primary examples:

| Audit | Observed | Gate | Result |
|---|---:|---:|---|
| maximum relative two-start delta gap | 0.0142552 | <= 0.01 | fail |
| maximum projected-gradient mapping | 2.30717e-5 | <= 1e-5 | fail |

The run was therefore correctly classified as `integrity_failure`; its router
decision was not promoted to a scientific conclusion.

The recovery changed only the deterministic convex-solver budget from 2,048
to 8,192 iterations. It did not change data, seeds, frozen checkpoints,
routers, scientific targets, success thresholds, or convergence tolerances.
It was written separately to `../formal_recovery/`.

Reproducibility checks between this directory and `formal_recovery/`:

- `subset_calibration.json` SHA-256:
  `00bc66648b68e15ebba09879eb8bf031739c624b71250189ead09638530a9f0f`;
- `router_training.json` SHA-256:
  `77391c0d0092f29f43fb7556c486846b55d0ca8266a5aed797414e264a8f7c07`;
- `router_checkpoint.pt` SHA-256:
  `1c86c707a3a1ba7a8bc5190a41d60d307ccf34e1d2f3d4b25c63a543e13a0465`;
- labels, competitors, memory predictions, population groups, and all
  non-simplex condition predictions are exactly equal.

The converged result and interpretation are in
`../formal_recovery/ANALYSIS.md`.
