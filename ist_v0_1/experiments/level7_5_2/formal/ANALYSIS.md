# Level 7.5.2 formal analysis

## Outcome

The formal run completed with integrity **PASS** and the registered
classification:

`weak_L2_precursor_partially_replicated`

One of the two preregistered seed1879 positive checkpoints replicated the
complete mirrored weak-L2 rule: step1400 passed, while step1600 failed. None of
the nine default-L3 controls produced an L2 false positive, and all five
registered L3 calibration positives were redetected. The formal outcome is
therefore partial replication, not full confirmation and not a null result.

The run completed all 16 frozen milestones and all 256 milestone-condition
panels in 14,214.4 seconds (3 h 56 min 54 s).

## Integrity

Every registered gate passed:

- the frozen Level 7.5.1 result hash and classification matched;
- all four parent exact-replay gates remained passed;
- all 16 checkpoint hashes and sizes matched their registration;
- 16/16 milestones and all 16 conditions per milestone completed at N=4,096;
- every model remained frozen and its fingerprint remained unchanged;
- the new shared dataset seed was `7520000`, distinct from the discovery panel
  seed `7510000`;
- no training, checkpoint mutation, sample extension, or seed replacement
  occurred.

The result is therefore interpretable as an independent evaluation-panel test
on the same frozen training trajectories.

## Registered primary test

| Component | Required | Observed | Result |
|---|---:|---:|:---:|
| seed1879 registered L2 positives | 2/2 | 1/2 | partial |
| default-L3 L2 false positives | 0/9 | 0/9 | pass |
| secondary L3 calibration | 5/5 | 5/5 | pass |

The registered positives behaved as follows:

| Step | Weak L2 | Intact | `zero_L2` | `roll_L2` | `keep_L2` | L2 keep margin |
|---:|:---:|---:|---:|---:|---:|---:|
| 1400 | pass | 26.61% | 5.52% | 9.89% | 25.34% | +19.70 pp |
| 1600 | fail | 34.94% | 7.45% | 10.30% | 29.98% | +22.22 pp |

Step1600 passed every registered weak-L2 clause except
`other_layer_zeros_preserve_intact`. Its preservation floor was 29.94%, but
`zero_L3` reached 28.32%, missing the rule by 1.62 percentage points. In more
direct causal terms, removing L3 reduced query accuracy by 6.62 points, just
beyond the frozen maximum permitted drop of 5 points.

This failure is not evidence that L2 ceased to be the core at step1600:

- zeroing L2 reduced 34.94% intact behavior to 7.45%;
- rolling L2 reduced it to 10.30%;
- retaining only L2 preserved 29.98%;
- retaining only L3 preserved only 7.76%;
- the L2 single-layer retention advantage was +22.22 points.

The failed clause instead detects that L3 had already become a causally useful
support layer.

## Independent window result

Step1300, which was frozen as descriptive window context rather than as a
registered primary positive, crossed the complete weak-L2 rule on the new
panel. The independent panel therefore placed the exact threshold-defined L2
window at steps1300-1400 rather than reproducing the discovery panel's
step1400/1600 pair.

| Step | Weak L2 | Intact | `keep_L2` | `keep_L3` | L3 removal drop | Main failed clause when negative |
|---:|:---:|---:|---:|---:|---:|---|
| 1200 | no | 12.28% | 10.99% | 5.62% | 1.81 pp | behavior/retention below 20% |
| 1300 | yes | 23.00% | 21.24% | 5.54% | 1.64 pp | — |
| 1400 | yes | 26.61% | 25.34% | 5.47% | 2.93 pp | — |
| 1500 | no | 30.49% | 22.71% | 5.54% | 7.10 pp | L3 removal exceeds 5 pp |
| 1600 | no | 34.94% | 29.98% | 7.76% | 6.62 pp | L3 removal exceeds 5 pp |
| 1700 | no | 37.82% | 30.52% | 12.96% | 7.13 pp | L3 removal exceeds 5 pp |
| 1800 | no | 28.08% | 20.09% | 13.38% | 7.15 pp | L3 support plus selectivity loss |

The new panel suggests a coherent within-C2 sequence:

1. step1200: long-context behavior is still below the weak formation floor;
2. steps1300-1400: a relatively pure L2-selective scaffold is measurable;
3. from step1500: L2 remains dominant, but removing L3 now costs more than
   five points;
4. by steps1700-1800: L3-only retention rises while L2 remains the stronger
   single layer, producing a proto-L2-core/L3-support topology.

This topology anticipates the stable seed1879 circuit observed later. It does
not mean the full final route was already complete in C2: intact performance
remained far below the strict 90% behavior criterion.

## Discovery-panel comparison

The exact threshold membership moved at two boundary-sensitive checkpoints:

| Step | Discovery N=1,024 | Independent N=4,096 | L3 removal drop: old → new |
|---:|:---:|:---:|---:|
| 1300 | fail | pass | 2.25 → 1.64 pp |
| 1400 | pass | pass | 2.25 → 2.93 pp |
| 1600 | pass | fail | 4.88 → 6.62 pp |

Step1400 is the direct out-of-sample replication common to both panels.
Step1300 crossed the 20% retention/preservation floors on the larger panel;
step1600 crossed the opposite boundary because its L3-removal drop moved from
just below to above the fixed 5-point allowance. The correct preregistered
classification remains partial replication. The neighboring switches must not
be used to redefine the primary positive set after the fact.

## Route specificity and calibration

All nine checkpoints from seeds2203, 2551, and 2909 remained negative under
the mirrored weak-L2 rule. Conversely, the independent panel redetected all
five known weak-L3 calibration states:

- seed2203 step1000;
- seed2551 steps700 and800;
- seed2909 steps700 and800.

Thus the L2 signal is not a generic consequence of weak long-context behavior
or of the intervention implementation. On the same new examples and with the
same thresholds, the exceptional trajectory selects L2 while the three
default trajectories select L3.

## Conclusions and limits

Level 7.5.2 establishes three things with different strengths:

1. **Formal:** the exact 2-of-2 primary claim partially replicated (1/2), so
   full independent confirmation was not achieved.
2. **Strong route-specific evidence:** seed1879 contains a reproducible weak-L2
   state at step1400, an additional L2-positive state at step1300, and zero L2
   false positives among nine default-route controls.
3. **New mechanistic hypothesis:** the loss of pure-L2 rule membership after
   step1400 is explained by recruitment of L3 support rather than disappearance
   of L2 dominance.

The experiment uses a new dataset but not new training initializations. The
candidate window was selected after Level 7.5.1, the rule uses hard point
thresholds, and the new L2-to-L3-support interpretation is secondary. It does
not yet prove that the early L2 state causes the eventual route.

## Next experiment

Do not continue searching evaluation thresholds. Level 7.5.3 should test route
commitment causally with preregistered training-time counterfactual branches:

- exactly replay seed1879 from before step1200;
- apply a fixed transient L2-Memory suppression over the registered commitment
  window, with intact replay and matched L3 suppression as controls;
- release the intervention, continue the unchanged C2/C4 schedule, and classify
  the final route on one new frozen causal panel;
- use matched L3-route branches to test the converse layer specificity.

The decisive question is whether disrupting the early selected layer changes
or prevents the later route, rather than whether another nearby checkpoint
crosses the same descriptive threshold.

## Artifacts

- `result.json`: full N=4,096 metrics, rule components, audits, and diagnosis
- `summary.json`: compact milestone profiles
- `progress.json`: completion and registered classification
- `independent_weak_L2_confirmation.png`: target and control trajectories
- `seed*/step_*/`: per-condition resumable metrics and milestone results

