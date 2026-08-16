# IST persistent-Memory evidence ledger

This ledger is the compact public-facing map of the frozen mechanism chain. The
machine-readable source of truth is `claim_registry.json`; exact estimates,
confidence intervals, and limitations remain in each linked formal analysis.

| ID | Frozen conclusion | Status | Authoritative stage |
|---|---|---|---|
| selective_pollution_defense | Frozen single-step risk-utility intervention produced a small clean-safe aggregate recovery under Memory pollution. | Supported | Level 6.16.2 |
| output_head_rescue | Replacing only the token output head causally rescued the seed707 12-chunk gate while Memory remained necessary. | Supported | Level 6.18.3 |
| task_aligned_context_subspace | A task-aligned Memory-context subspace beat rolled and random controls. | Supported | Level 6.18.8 |
| hard_example_read_access_failure | On source errors, information was frequently present in final-layer Memory but lost through deployed read access. | Supported boundary | Level 6.19 |
| signed_affine_simplex_obstruction | Head-budget reallocation was insufficient; signed affine value mixing closed the registered tangent gap. | Supported boundary | Level 6.19.3 |
| oracle_not_compiled | The causal Oracle mechanism did not compile into the frozen label-free router. | Registered negative | Level 6.19.4 recovery |
| joint_calibration_coupling_bottleneck | Dose and signed direction were separately observable, while joint deployment calibration remained limiting. | Supported boundary | Level 6.19.5 |
| factorized_repair_branch_closed | The sole factorized repair missed its conjunction, so the router-repair branch closed. | Registered negative | Level 6.19.6 |

## Quantitative anchors

- Level 6.16.2: overall pollution-policy gain `+0.52 pp`, bootstrap 95% CI
  `[+0.23, +0.80] pp`; maximum clean trigger rate `4.5%`.
- Level 6.18.3: 12-chunk accuracy `89.99% -> 96.19%` and 16-chunk accuracy
  `83.25% -> 91.31%` when only the output head changed.
- Level 6.19: on 277 source errors, final-layer Memory probe accuracy was
  `81.95%`, while deployed behavior was wrong by construction on the error
  subset.
- Level 6.19.5: probed-dose plus Oracle-direction recovery was `94.38%`, but
  the frozen label-free router recovered only `21.35%`.
- Level 6.19.6: the single factorized repair recovered `24.41%`, below the
  `25%` threshold, and passed only four of six specificity contrasts.

## Claim boundary

These experiments support persistent, distributed, causally usable Memory in
the tested synthetic cross-chunk task and identify concrete readout and
pollution mechanisms. They do not establish universal superiority over a
standard Transformer, real-world generalization, or a production-ready router.
Positive mechanism findings and registered negative results are both retained;
near-threshold failures are not rounded into successes.
