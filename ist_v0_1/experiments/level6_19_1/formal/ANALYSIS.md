# Level 6.19.1 formal analysis

## Decision

**Level 6.19.1 passes every integrity check but does not causally validate the
Level 6.19 linear-Probe slot ranking as a sufficient read mechanism.** The
registered classification is `linear_slot_ranking_not_causally_sufficient`.

On 4,096 new 16-chunk examples, the frozen Level 6.18.3 source is correct on
3,783/4,096 = **92.36%** and makes 313 errors. The frozen Level 6.19 Memory
Probe is correct on 260/313 = **83.07%** of those errors; these 260
Memory-decodable errors form the registered primary population.

The top-four, 4x-odds intervention delivers a large and exact routing dose: its
selected-slot attention mass rises from 11.82% to 28.53%. It increases the
registered context-Probe correct-versus-source-rival margin by +0.0516, but it
does not pass all equal-dose specificity comparisons and corrects only 2/260
Memory-decodable errors. Across the complete panel those two corrections are
exactly cancelled by two regressions.

By contrast, the equal-L2 context-gradient positive control corrects 20/260
Memory-decodable errors and 21/313 source errors, with no regression anywhere
in the 4,096-example panel. The downstream path can therefore use an improved
context direction; the failed component is the linear Probe's slot-ranking
intervention, not the existence of any usable downstream direction.

The registered next boundary is:

> The downstream path responds to equal-dose context improvement, but linear-
> Probe slot targeting is not a sufficient read mechanism.

No deployable checkpoint is produced. All interventions are label-aware frozen
mechanism tests.

## Integrity

Every registered implementation and boundary check passed:

- source checkpoint: formally passed Level 6.18.3 seed707 at 16 chunks;
- failed Level 6.18.9 candidate excluded;
- model, original Memory Probe, and Level 6.19 probes fully frozen;
- model and all Probe state fingerprints bitwise unchanged after the run;
- returned persistent Memory exactly invariant under every condition;
- gradient-leaf forward reconstructs source logits exactly, maximum error
  `0.0`;
- repeated unmodified source forward reconstructs source logits exactly,
  maximum error `0.0`;
- protected tests unopened and seed909 locked.

The formal dataset seed is disjoint from Level 6.19 and from the two smoke-only
panels. The 313 source errors exceed the registered minimum of 200, and the 260
Memory-decodable errors exceed the registered minimum of 150.

The confidence-matched control contains 313 source-correct examples selected
without replacement. Mean source confidence is 1.3858 on errors and 1.4557 on
matched correct examples. Mean absolute matching distance is 0.0702 and the
maximum is 0.2188.

## Targeting manipulation check

On the 260 primary cases, top-four attention mass changes as follows:

| Condition | Selected top-four attention mass |
|---|---:|
| Source | 11.82% |
| Top-four 4x odds | 28.53% |
| Paired change | **+16.71 pp** [16.24, 17.20] |

All 260 examples receive a positive attention-mass change; sign-flip
`p=0.00010`. Thus the null result cannot be attributed to a missing or
ineffective attention bias. This check establishes delivered dose, not task
benefit.

## Primary Memory-decodable error panel

| Condition | Context decoder accuracy | Context margin | Deployed accuracy | Deployed margin | Context L2 dose |
|---|---:|---:|---:|---:|---:|
| source | 45.38% | 0.6437 | 0.00% | -1.8650 | 0.0000 |
| top-four 2x | 43.08% | 0.6684 | 0.00% | -1.8666 | 0.3687 |
| top-four 4x | 45.38% | 0.6953 | 0.77% | -1.8699 | 0.8468 |
| top-four 8x | 41.92% | 0.7189 | 1.15% | -1.8751 | 1.4119 |
| bottom-four 4x | 45.00% | 0.6355 | 0.38% | -1.8612 | 0.8485 |
| rolled top-four 4x | 44.23% | 0.6803 | 0.00% | -1.8679 | 0.9241 |
| random four 4x mean | 45.29% | 0.6453 | 0.10% | -1.8634 | 0.5630 |
| context-gradient positive control | 47.31% | 0.6679 | **7.69%** | **-1.6942** | 0.8468 |

The top-four dose curve cleanly separates the independent Probe objective from
deployed behavior. As odds rise from 2x to 4x to 8x, context-Probe margin change
is approximately +0.0247, +0.0516, and +0.0752. Over the same doses, deployed
correct-versus-rival margin change is -0.0016, -0.0049, and -0.0102. Increasing
the registered routing dose strengthens the Probe direction while moving the
frozen deployed decision slightly in the wrong direction.

Discrete context decoding also does not improve. At the registered 4x dose,
12 primary cases become context-decoder correct and 12 become wrong, for zero
net change. At 8x, 10 are corrected and 19 regress. The higher continuous
correct-versus-fixed-rival Probe margin therefore does not translate into
better full 16-class context decoding.

## Registered specificity families

### Context-Probe margin

| Registered contrast on 260 primary cases | Estimate | 95% CI | Raw p | Holm p | Pass |
|---|---:|---:|---:|---:|---:|
| top-four 4x vs source | +0.0516 | [+0.0128, +0.0885] | 0.0093 | 0.0372 | yes |
| top-four 4x vs random mean | +0.0500 | [+0.0109, +0.0901] | 0.0163 | 0.0489 | yes |
| top-four 4x vs bottom-four 4x | +0.0598 | [+0.0000, +0.1213] | 0.0521 | 0.1042 | no |
| top-four 4x vs rolled top-four 4x | +0.0150 | [-0.0288, +0.0568] | 0.5070 | 0.5070 | no |

The intervention has a reproducible Probe-margin signal relative to source and
random slots, but it fails the preregistered requirement that all four
equal-dose contrasts survive Holm correction. In particular, another example's
top-four selection performs similarly.

The top-four selection is partly global rather than strongly example-specific.
Slots 11, 2, 0, 17, and 7 appear among the selected four on 48.1%, 46.5%,
46.2%, 45.8%, and 42.7% of primary examples. A rolled example's selection
shares 1.22 of four slots on average with the receiver selection. This overlap
helps explain why the rolled control is demanding, but it does not change the
registered failure: the current scoring rule does not demonstrate selective
per-example routing specificity.

### Deployed correctness

| Registered contrast on 260 primary cases | Estimate | 95% CI | Holm p | Pass |
|---|---:|---:|---:|---:|
| top-four 4x vs source | +0.77 pp | [0.00, +1.92] | 1.0 | no |
| top-four 4x vs random mean | +0.67 pp | [0.00, +1.73] | 1.0 | no |
| top-four 4x vs bottom-four 4x | +0.38 pp | [-0.77, +1.92] | 1.0 | no |
| top-four 4x vs rolled top-four 4x | +0.77 pp | [0.00, +1.92] | 1.0 | no |

Only two primary errors are corrected by top-four 4x. The discrete sign-flip
family is consequently far from confirmation. None of the four registered
behavioral specificity tests passes.

## Full-panel retention and controls

The registered 4x intervention passes its safety gates, but only because its
behavioral effect is almost zero:

| Population | Source | Top-four 4x | Change, 95% CI | Corrections | Regressions |
|---|---:|---:|---:|---:|---:|
| all 4,096 | 92.36% | 92.36% | 0.00 pp [-0.10, +0.10] | 2 | 2 |
| 313 matched correct | 100.00% | 99.36% | -0.64 pp [-1.60, 0.00] | 0 | 2 |
| all 3,783 source-correct | 100.00% | 99.95% | -0.05 pp [-0.13, 0.00] | 0 | 2 |
| all 313 source errors | 0.00% | 0.64% | +0.64 pp [0.00, +1.60] | 2 | 0 |

The full-panel lower confidence bound remains above the registered -1-point
non-inferiority margin, and the matched-correct lower bound remains above the
registered -2-point margin. These passes establish lack of large harm, not
useful rescue.

Across the full panel, top-four 2x has zero corrections and two regressions;
4x has two of each; and 8x has three corrections and four regressions. The
random controls range from one net regression to one net correction. The main
condition is not behaviorally distinguishable from this background switching.

The top-four 4x Probe margin rises by +0.0261 across all 4,096 examples, but
full-panel context-decoder accuracy falls from 91.58% to 90.94%, a significant
-0.63-point change [ -1.07, -0.17 ]. Again, a higher fixed-rival linear-Probe
margin is not equivalent to better categorical information or behavior.

## Context-gradient positive control

The positive control moves each final-query context along the exact frozen
correct-versus-deployed-rival gradient, with per-example L2 norm matched to the
top-four 4x context change. On the primary group, both interventions therefore
have mean context dose **0.8468**.

Their downstream effects are sharply different:

| Equal-L2 intervention | Deployed margin change | Corrected primary errors |
|---|---:|---:|
| top-four 4x routing | -0.0049 [-0.0109, +0.0015] | 2/260 |
| context-gradient positive control | **+0.1708** [+0.1551, +0.1886] | **20/260** |

Every primary example receives a positive deployed-margin change from the
gradient control (`p=0.00010`). Its 20 corrections and zero regressions give
McNemar `p=1.91e-6`. On all 313 source errors it corrects 21/313 = 6.71%; on the
complete panel it raises accuracy from 92.36% to **92.87%**, exactly 21 net
corrections with no regressions.

This is a local, true-label-aware oracle. It is neither a fair deployable
baseline nor an authorized checkpoint. Its purpose is to show that an
equal-sized context change can pass through the frozen fusion, FFN/residual,
normalization, and output path and improve actual decisions. The downstream
path is operational; the top-four routing change points in the wrong deployed
subspace.

The positive control raises context-Probe margin by only +0.0242 on primary
cases, less than the top-four intervention's +0.0516, while its deployed-margin
gain is more than 0.17 rather than slightly negative. This inversion is the
strongest evidence that the independent context Probe and the deployed model
use different task directions.

## Scientific conclusion

Level 6.19.1 refines the Level 6.19 access diagnosis:

1. Persistent Memory remains highly informative on source errors: the frozen
   Memory Probe succeeds on 83.07% in this new panel.
2. Attention to the registered high-contribution slots can be strongly and
   cleanly increased without changing persistent Memory.
3. This routing change increases the exact metric used to rank those slots,
   especially at higher doses.
4. It does not improve categorical context decoding or deployed decisions and
   is not specific to the per-example ranking under all registered controls.
5. An equal-norm deployed-gradient context direction produces a large,
   regression-free behavioral effect, proving that the frozen downstream path
   can use a better-aligned context.

The supported claim is therefore narrower than “the model merely attends to
the wrong four slots.” Correct-label information is present in persistent
Memory, but the linear Probe's additive per-slot decomposition does not specify
how the multi-head read must compose that information into the deployed context
subspace. The missing mechanism may involve head-specific key routing,
value-projection geometry, interactions among slots, or a context direction
that cannot be reached by simple positive reweighting of existing attention.

The result does not contradict the Level 6.19 localization. It rejects one
simple causal implementation of the read-access hypothesis. It also explains
why another round of top-k selection, a stronger odds multiplier, or ordinary
`memory_read` optimizer search is not the next justified step.

## Next experiment

The next milestone should be **Level 6.19.2: frozen read-attention reachable-
subspace audit**.

For each Memory-decodable source error, decompose the final multi-head read and
ask whether the successful context-gradient direction is geometrically
reachable by reweighting the existing 32 slot value projections:

1. compute the exact deployed-margin gradient with respect to each head's
   attention logits and projected slot values;
2. solve a frozen, per-example attention-simplex/KL-budget oracle that maximizes
   the first-order deployed margin without changing keys, values, or model
   parameters;
3. compare its achievable context direction and corrections with top-four
   Probe routing, gradient-ranked routing, random controls, and the unrestricted
   context-gradient positive control at matched L2/KL doses;
4. separate head-specific routing limitation from value-projection/subspace
   limitation;
5. retain full-panel and matched-correct regression audits, while keeping
   seed909 and protected tests locked.

If the constrained attention oracle approaches the context positive control,
the router score—not the read architecture—is the obstruction. If even the
oracle cannot project onto the successful context direction, the next causal
boundary is value composition/output projection rather than attention slot
selection.
