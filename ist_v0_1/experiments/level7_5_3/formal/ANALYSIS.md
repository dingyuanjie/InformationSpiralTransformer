# Level 7.5.3 formal analysis

## Outcome

The formal run completed with integrity **PASS** and the registered
classification:

`transient_suppression_disrupts_routes_nonspecifically`

The preregistered layer-specific route-commitment hypothesis was not
confirmed. Selected-layer suppression changed the registered endpoint route
class in only 2/4 seeds, while matched other-layer suppression changed it in
4/4. Seed1879 showed no L2-specific effect, and none of the three default L3
seeds met the complete directional criterion.

The run completed all 12 fixed-compute training branches and all 12 N=2,048
endpoint panels in 6,845.5 seconds (1 h 54 min 06 s).

## Integrity and exact controls

All registered integrity conditions passed:

- every fixed-stage, C2, and C4 source hash matched;
- all four intact branches exactly reproduced the original C2 and C4 model,
  Probe, optimizer, CPU/CUDA RNG, validation history, and stop steps;
- all 12 branches completed their fixed source-specific C2/C4 budgets;
- each intervention branch contained exactly 200 suppressed C2 updates;
- no suppression occurred during C4 and no branch trained beyond its frozen
  C4 endpoint;
- all 12 endpoint models remained frozen during the causal panel;
- every endpoint completed all 16 conditions at N=2,048 on shared dataset seed
  `7530000`;
- seed909 was not used.

The negative result therefore cannot be attributed to approximate replay,
branch compute mismatch, incomplete panels, or a failed intervention count.

## Registered endpoint result

| Seed | Original route | Intact replay | Selected layer suppressed | Other layer suppressed | Selected drop | Other drop | Specific effect |
|---:|---|---|---|---|---:|---:|:---:|
| 1879 | L2 core/L3 support | 94.73%, original | 83.59%, unformed | 82.76%, unformed | 11.13 pp | 11.96 pp | no |
| 2203 | L3 core | 97.12%, original | 77.59%, unformed | 86.18%, unformed | 19.53 pp | 10.94 pp | no |
| 2551 | L3 core | 96.44%, original | 93.02%, L3 core | 67.58%, unformed | 3.42 pp | 28.86 pp | no |
| 2909 | L3 core | 91.80%, original | 92.63%, L3 core | 89.89%, unformed | -0.83 pp | 1.90 pp | no |

For seed1879, suppressing L2 and suppressing L3 caused similarly sized losses;
the matched L3 control was slightly more damaging. For seed2203, L3
suppression had the larger effect, but L2 suppression also prevented the
endpoint from reaching the formation threshold. Seeds2551 and2909 showed the
opposite of the registered prediction: transient L3 suppression recovered the
original L3 route, while transient L2 suppression produced the weaker final
endpoint.

The seed2909 L2-suppressed endpoint was especially close to the hard boundary:
89.89%, only 0.11 percentage points below the registered 90% formation floor.
It must remain classified as unformed under the frozen point-estimate rule.

## No observed L2-to-L3 or L3-to-L2 rerouting

The registered classifier calls every below-90% endpoint
`unformed_behavior`, so six branches formally changed route class. However,
none switched to the opposite layer organization.

Across all 12 endpoints, the dominant single-layer retention identity remained
the same as the source trajectory:

- all three seed1879 branches retained more behavior with L2 alone than with
  L3 alone;
- all nine default-seed branches retained more behavior with L3 alone than
  with L2 alone.

Examples:

| Seed/branch | Intact | Keep L2 | Keep L3 | Dominant identity |
|---|---:|---:|---:|---|
| 1879 / intact | 94.73% | 88.92% | 12.45% | L2 |
| 1879 / L2 suppressed | 83.59% | 56.10% | 19.73% | L2 |
| 1879 / L3 suppressed | 82.76% | 77.64% | 12.45% | L2 |
| 2203 / L3 suppressed | 77.59% | 5.37% | 77.00% | L3 |
| 2551 / L2 suppressed | 67.58% | 6.64% | 67.14% | L3 |
| 2909 / L2 suppressed | 89.89% | 5.96% | 89.70% | L3 |

Thus the intervention altered the strength or completion of long-context
formation under the fixed budget, not the identity of the eventual carrier
layer. Level 7.5.3 supplies no evidence that a transient mask can redirect an
L2 trajectory into L3 or an L3 trajectory into L2.

## Training-range recovery versus long-context extrapolation

Every intervention was released before C4. By the frozen C4 endpoint, all 12
branches scored 96.25%-100% on the in-training four-chunk validation and their
Probe minima were 97.5%-100%. Nevertheless, fresh 16-chunk performance ranged
from 67.58% to 93.02% in the intervention branches.

This separation is important: the masks did not simply prevent the model from
learning the C4 task. The branches recovered short-range training competence,
but differed substantially in extrapolation to 16 chunks. The causal effect is
therefore better described as a change in long-context formation efficiency or
generalization margin than as a clean switch of route identity.

The C2 trajectories also show recovery after mask release. For example:

- seed2551 L3 suppression reduced the window-end C2 query to 40%, but it
  recovered to 97.5% by the original C2 endpoint and retained the L3 route at
  C4;
- seed2909 L3 suppression reduced the window-end query to 65%, then recovered
  to 100% at C2 and C4;
- seed1879 L2 suppression reduced the window-end query from the intact 81.2%
  to 42.5%, but its in-range C4 query later reached 98.8%.

Transient disruption can therefore be compensated after release, although the
amount of 16-chunk recovery differs by initialization and suppressed layer.

## Mechanistic update

The combined evidence now supports a more limited model:

1. Early layer-selective signatures predict the later carrier topology.
2. Those signatures are not uniquely necessary commitment switches under a
   200-step zero-mask intervention.
3. Route identity is robust: all branches return to the same dominant layer
   organization rather than selecting the other route.
4. Multiple layers contribute to optimization and long-context formation even
   when only one layer becomes the final dominant carrier.
5. L2 may act as an optimization scaffold in some default L3 trajectories:
   suppressing L2 was substantially more damaging than suppressing L3 in
   seed2551 and modestly more damaging in seed2909.

The last point is a secondary hypothesis, not a registered finding. Seed2203
shows the opposite sensitivity, so it is not universal.

## Limits

The intervention simultaneously changes three coupled mechanisms: recurrent
forward information, cross-chunk gradient flow, and the layer contribution
read by the Probe loss. Zero masking also creates a strong training
distribution shift. The matched-layer controls reveal that this disruption is
not route-specific, but they do not identify which of those mechanisms causes
the long-context deficit.

Route class also contains a hard 90% formation gate. A branch can preserve the
same layer topology yet be labeled unformed, as occurred here. The formal
classification remains correct under the preregistration, while the topology
analysis explains what that classification does and does not mean.

## Next experiment

The next step should be **Level 7.5.3.1: unsuppressed recovery dynamics**, not
another threshold search. Resume all 12 frozen C4 endpoints with no masks and
a common additional C4 budget, evaluate at fixed recovery milestones, and ask
whether the six below-threshold endpoints converge back to their original
route and 16-chunk performance.

- Recovery would show that the 200-step intervention mainly delays
  long-context formation.
- A stable deficit with preserved layer identity would indicate persistent
  damage to generalization strength rather than rerouting.
- Only a later change in dominant layer identity would support genuine route
  migration.

After recovery is resolved, a separate factorial intervention can decompose
recurrent-state masking, gradient detachment, and Probe-only masking.

## Artifacts

- `result.json`: full replay gates, training branches, endpoint panels, and
  registered diagnosis
- `summary.json`: compact routes and effect sizes
- `progress.json`: completed branch/panel counts
- `route_commitment_counterfactual.png`: endpoint behavior and route classes
- `seed*/<branch>/`: resumable checkpoints, histories, exact gates, and causal
  metrics

