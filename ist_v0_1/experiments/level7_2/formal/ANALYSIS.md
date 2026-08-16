# Level 7.2 formal analysis

## Decision

- Run integrity: **PASS**.
- Seed1601: **curriculum failure** at 4 chunks; candidate selection and
  protected testing were not opened.
- Seed1879 checkpoint selection: **PASS**; all four candidates were eligible
  and the registered tie rule selected zero-Probe step 750.
- Seed1879 one-time protected behavior: **PASS**, 97.0947% query accuracy on
  4,096 samples (95% Wilson CI [96.5346%, 97.5666%]).
- Seed1879 all-Memory causal necessity: **PASS descriptively**; reset, zero,
  and batch roll reduced query accuracy to approximately chance.
- Seed1879 registered final-layer causal gate: **FAIL**; final-layer Memory was
  neither necessary nor sufficient under the preregistered thresholds.
- Registered classification: `causal_gate_failed`.
- Strong or conditional stable causal formation: **NOT SUPPORTED** under the
  complete registered conjunction.

The run completed normally. The negative classification is caused by a
specific causal-localization failure, not a program error and not a failure of
the one-time protected behavior panel.

## Integrity

The run used exactly the preregistered new seeds 1601 and 1879 and candidate
steps 300, 450, 600, and 750. Training settings remained identical to Level
7.1; saving additional candidate files did not alter training computation or
RNG consumption. No old seed was rescued, no third seed or additional
candidate was added, seed909 remained closed, and neither an output-head nor a
router repair was used.

The selected checkpoint was chosen only from the 1,024-example validation
panel. The protected panel was stored immediately after its single opening and
was not used for selection or retraining. The seed1879 model fingerprint was
identical before and after protected/causal evaluation:

`ad113264a0227bb69770dc56b7f395b0bf999e7b4eb8c487408d07100043f77c`

All model parameters remained frozen during formal evaluation. Recorded source
hashes match the current frozen files:

- runner SHA-256:
  `17be114c7c067a1dda7d3d8e8303a656bedc22a84e3fe488ddfc999afce7c941`;
- static preregistration SHA-256:
  `f93eaeb200538a138c72decd8538071017c2cf9f263d6a71aae0dadaaaee90eb`.

Aggregate integrity: **PASS**.

## Unchanged training trajectory

| Seed | Fixed stage | Curriculum stages | Candidate trajectory reached | Training time |
|---:|---:|---|---|---:|
| 1601 | PASS at 1,800 steps | 2 chunks: PASS at 1,200; 4 chunks: **FAIL at 1,500** | No | 6.73 min |
| 1879 | PASS at 1,600 steps | 2/4/8/16 chunks: PASS at 2,300/1,000/100/300 | Yes | 20.10 min |

Seed1601 ended its 4-chunk budget at 83.75% query, 91.25% local, and 75.00%
minimum Probe accuracy. Its failure precedes the new retention-selection
hypothesis, so no candidate, protected test, or causal panel was opened for
that seed.

Seed1879 entered zero-Probe maintenance after a 96.25% 16-chunk curriculum
evaluation with 95.00% minimum Probe accuracy. It completed the unchanged
maintenance budget and produced all four registered candidates.

Formation reliability therefore remained heterogeneous: only one of two new
initializations reached the checkpoint-selection stage.

## Validation-only checkpoint selection

Every seed1879 candidate was evaluated on the same frozen 1,024-example
validation panel.

| Zero-Probe step | Query | 95% Wilson CI | Local | Eligible |
|---:|---:|---:|---:|---:|
| 300 | 96.3867% | [95.0594%, 97.3673%] | 99.4141% | Yes |
| 450 | 96.5820% | [95.2836%, 97.5322%] | 99.4141% | Yes |
| 600 | 96.4844% | [95.1714%, 97.4499%] | 99.4141% | Yes |
| 750 | **96.5820%** | **[95.2836%, 97.5322%]** | 99.4141% | Yes |

Steps 450 and 750 tied on validation query accuracy. The preregistered
secondary ranking preferred the later zero-Probe step, so step 750 was selected
without consulting protected data.

The candidate curve is nearly flat and every point is eligible. Consequently,
this run establishes stable behavior along the maintenance tail for seed1879,
but it does **not** show that selecting an earlier checkpoint rescues an
otherwise bad final endpoint. The selected candidate is the original final
750-step endpoint. Checkpoint-endpoint instability is therefore not a general
explanation of the Level 7.1 failures.

## One-time protected behavior

The single selected seed1879 checkpoint was evaluated once on the protected
4,096-example panel:

- query: 3,977/4,096 = **97.0947%**;
- query 95% Wilson CI: **[96.5346%, 97.5666%]**;
- local: 4,085/4,096 = **99.7314%**;
- local 95% Wilson CI: **[99.5197%, 99.8500%]**.

Both registered behavior thresholds passed. This is positive independent
evidence that a new initialization can retain high 16-chunk behavior after the
complete zero-Probe schedule. It does not by itself satisfy the full Level 7.2
mechanistic conjunction.

## Conditional causal audit

The protected behavior pass authorized the separate 1,024-example causal
panel.

| Condition | Query | Change from intact | Local | Registered role |
|---|---:|---:|---:|---|
| Intact | **97.7539%** | — | 99.9023% | behavior reference |
| Reset all Memory | 6.1523% | -91.6016 pp | 99.9023% | all-Memory disruption |
| Zero all Memory | 6.1523% | -91.6016 pp | 99.9023% | all-Memory disruption |
| Batch-roll all Memory | 6.2500% | -91.5039 pp | 99.9023% | all-Memory disruption |
| Zero final-layer Memory | **91.6016%** | -6.1523 pp | 99.9023% | final-layer necessity |
| Batch-roll final-layer Memory | **86.0352%** | -11.7188 pp | 99.9023% | final-layer alignment |
| Keep only final-layer Memory | **11.9141%** | -85.8398 pp | 99.9023% | final-layer sufficiency |

Three conclusions must be separated:

1. **Persistent Memory is causally necessary.** Destroying, resetting, or
   assigning another sample's complete Memory reduced query behavior from
   97.75% to 6.15–6.25%, while the local task stayed at 99.90%.
2. **Final-layer Memory contributes but is not necessary.** Zeroing it caused
   only a 6.15-point loss and left 91.60% query accuracy. Batch rolling caused
   a larger but still non-catastrophic 11.72-point loss.
3. **Final-layer Memory is not sufficient.** Keeping only that layer produced
   11.91%, far below the registered 90% sufficiency threshold.

The formal causal gate required final-layer disruptions to be <=20% and the
keep-final-layer condition to be >=90%. It therefore fails decisively. The
aggregate `max_disrupted_query` was 91.6016%, and keep-L3 was 11.9141%.

## Scientific conclusion

Level 7.2 provides both a positive replication and a negative scope correction:

- a new seed retained 97.09% protected 16-chunk behavior after full zero-Probe
  maintenance;
- complete persistent Memory remained overwhelmingly causal on a fresh panel;
- the earlier claim that the final Memory layer is necessary and sufficient did
  not generalize to seed1879;
- layer localization is therefore initialization-dependent: this model uses a
  cross-layer persistent-Memory circuit rather than the previously observed
  final-layer-dominant circuit;
- the training protocol remains unreliable unconditionally because seed1601
  failed at only four chunks;
- validation checkpoint selection was not shown to outperform the fixed final
  endpoint because step 750 itself was selected and generalized.

The project-level Memory claim is strengthened: high long-range behavior in a
new initialization collapses when all persistent Memory is disrupted. The
layer-specific claim must be weakened: final-layer necessity and sufficiency
describe some successfully formed checkpoints, not a universal IST mechanism.

The official Level 7.2 classification remains `causal_gate_failed`. It cannot
be upgraded to conditional stable causal formation by dropping the failed
layer-localization clauses after seeing the causal panel.

## Registered stop boundary

Do not select a different candidate, evaluate another candidate on protected
data, rerun the protected panel, extend seed1601, add a third seed, alter the
causal thresholds, repair an output head/router, or open seed909. Level 7.2 is
closed under its registered classification.

A future layer-localization study must be separately preregistered and treat
cross-layer heterogeneity as the hypothesis. It may not relabel this full-gate
failure or use the Level 7.2 protected panel for selection.
