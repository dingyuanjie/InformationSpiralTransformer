# Level 7.4 formal analysis

## Decision

- Run integrity: **PASS**.
- Frozen checkpoints completed: **10/10**.
- Registered conditions completed: **11/11 at every checkpoint**.
- Total fixed evaluations: **110 checkpoint-condition cells**, each N=1,024.
- Registered trajectory classification:
  `l2_core_l3_support_established_by_16chunk_and_stable`.
- Earliest formed 16-chunk behavior: **curriculum_4**.
- Earliest L2-core/L3-support route: **curriculum_4**.
- Earliest stable L2-core/L3-support suffix: **curriculum_4**.
- Post-curriculum route transitions: **none**.

The primary preregistered hypothesis passes: every checkpoint from the end of
16-chunk curriculum through zero-Probe step750 has the same L2-core/L3-support
class. The secondary trajectory is stronger than required: the route is
already present at the end of 4-chunk training and remains stable thereafter.

The run completed normally in 1,746.61 seconds (29.11 minutes).

## Integrity and frozen timeline

The ten source checkpoint SHA-256 values matched the preregistration. Every
checkpoint loaded a distinct model fingerprint, and each fingerprint was
identical before and after its eleven interventions. All parameters were
frozen. No model, Probe, checkpoint, output head, or router was trained,
selected, added, or omitted.

All evaluations used the same new dataset seed 7,400,000. The panel was not
reused from Level 7.2, 7.3, or 7.3.1. Seed909 remained closed. The runner and
static preregistration hashes recorded at completion match the current files:

- runner SHA-256:
  `47c173ea615db96cfcdfd7093beeb01fb47e1f18592ab0de2982ea1ffe75fce4`;
- static preregistration SHA-256:
  `3ead0cc09a2d181f7afd27b9b6f62ba86ec1e6cd8438bd71086569615cfdacfd`.

Aggregate integrity: **PASS**.

## Behavioral and route trajectory

| Checkpoint | Training stage | Intact query | Local | Whole-Memory max disruption | Route class |
|---|---|---:|---:|---:|---|
| C2 | curriculum through 2 chunks | 20.51% | 97.66% | 5.86% | unformed behavior |
| C4 | curriculum through 4 chunks | **95.80%** | 99.22% | 6.64% | L2 core + L3 support |
| C8 | curriculum through 8 chunks | 97.75% | 99.61% | 6.35% | L2 core + L3 support |
| C16 | curriculum through 16 chunks | 97.36% | 99.71% | 6.54% | L2 core + L3 support |
| W0.2 | Probe weight 0.2 end | 97.27% | 99.71% | 6.64% | L2 core + L3 support |
| W0.1 | Probe weight 0.1 end | 96.97% | 99.71% | 6.74% | L2 core + L3 support |
| Z300 | zero-Probe step300 | 96.68% | 99.71% | 6.74% | L2 core + L3 support |
| Z450 | zero-Probe step450 | 97.27% | 99.71% | 6.45% | L2 core + L3 support |
| Z600 | zero-Probe step600 | 97.07% | 99.71% | 6.74% | L2 core + L3 support |
| Z750 | zero-Probe step750 | 96.78% | 99.71% | 6.64% | L2 core + L3 support |

There is one registered adjacent class transition:

`C2: unformed_behavior -> C4: l2_core_l3_supported`

No later class transition occurs.

## C2-to-C4 formation boundary

The C2 checkpoint has solved the local task but not the deployed 16-chunk
query. Its intact accuracy is 20.5078% (95% Wilson interval
[18.1472%, 23.0888%]), while reset, zero, and batch roll of all Memory yield
5.37-5.86%. This is descriptively consistent with an early causal Memory
signal, but it remains below the registered 90% formation gate and is not
assigned a route class.

At C4, intact 16-chunk accuracy jumps to 95.8008% (95% Wilson interval
[94.3915%, 96.8677%]). Complete-Memory interventions remain at 5.37-6.64%,
showing that the new behavior is causally carried by persistent Memory rather
than by local-task competence.

Because no intermediate checkpoint exists between the saved C2 and C4
endpoints, Level 7.4 localizes formation to that interval, not to a particular
training step inside the 1,000-step C4 curriculum stage.

## Length generalization

C4 is especially informative: the model had been trained through a maximum of
four chunks, but every checkpoint in this study was evaluated at sixteen
chunks. Its 95.80% fresh 16-chunk accuracy therefore demonstrates a fourfold
length extrapolation at the moment the registered route first becomes visible.

This is evidence that the C2-to-C4 stage learned a reusable recurrent Memory
procedure rather than memorizing only the trained horizon. Subsequent 8- and
16-chunk curriculum stages improve or maintain the score, but they are not
required for the first high-accuracy 16-chunk deployment in this seed.

This conclusion is about the synthetic task and seed1879. It does not by
itself establish arbitrary-length extrapolation or general-language behavior.

## Stable L2-core/L3-support circuit

| Checkpoint | Zero L2 | Roll L2 | Keep L2 | Zero L3 | Roll L3 | Keep L3 | Keep L2+L3 | L3 pair gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | 12.60% | 13.09% | 88.87% | 88.77% | 83.01% | 12.40% | 96.29% | 7.42 pp |
| C8 | 13.09% | 13.48% | 89.06% | 90.53% | 84.67% | 12.79% | 96.39% | 7.32 pp |
| C16 | 13.09% | 13.67% | 89.84% | 90.14% | 84.08% | 12.70% | 97.17% | 7.32 pp |
| W0.2 | 13.28% | 13.57% | 89.84% | 90.14% | 84.38% | 12.79% | 96.88% | 7.03 pp |
| W0.1 | 13.18% | 13.87% | 90.14% | 89.75% | 83.79% | 12.79% | 97.56% | 7.42 pp |
| Z300 | 12.99% | 13.48% | 90.23% | 89.45% | 83.69% | 12.89% | 97.27% | 7.03 pp |
| Z450 | 13.18% | 13.77% | 89.84% | 90.23% | 83.98% | 12.89% | 97.17% | 7.32 pp |
| Z600 | 13.18% | 13.57% | 90.23% | 89.84% | 83.79% | 12.99% | 97.46% | 7.23 pp |
| Z750 | 13.28% | 13.67% | 90.14% | 89.06% | 83.20% | 13.09% | 97.07% | 6.93 pp |

From C4 onward, every registered component stays in a narrow band:

- intact query: 95.80-97.75%;
- zero-L2: 12.60-13.28%;
- batch-roll-L2: 13.09-13.87%;
- keep-L2: 88.87-90.23%;
- zero-L3: 88.77-90.53%;
- batch-roll-L3: 83.01-84.67%;
- keep-L3: 12.40-13.09%;
- keep-L2+L3: 96.29-97.56%;
- L3 pair gain over L2 alone: 6.93-7.42 points.

The near-flat curves show that Level 7.3.1's L2-core/L3-support interpretation
is not a peculiarity of the final selected checkpoint. It is already present
at C4 and changes little through two later curriculum stages, Probe withdrawal,
and 750 zero-Probe updates.

## Withdrawal conclusion

Auxiliary-Probe withdrawal does not create the L2 route, move it from another
layer, or visibly erode it in seed1879. C16, W0.2, W0.1, Z300, Z450, Z600, and
Z750 all have the same route class and very similar causal effect sizes.

Thus the successful seed1879 endpoint is best understood as maintaining an
early-formed recurrent circuit. The withdrawal phase is a stability filter for
this trajectory, not the mechanism's formation phase.

This does not contradict Level 7.1's formation failures in other seeds.
Level 7.4 follows one successful seed retrospectively and cannot show that
withdrawal is harmless in every initialization.

## Scientific conclusion

Level 7.4 establishes four within-seed findings:

1. deployable 16-chunk behavior and the L2-core/L3-support causal signature
   appear between the saved C2 and C4 curriculum endpoints;
2. a model trained through four chunks already generalizes the mechanism to a
   fresh sixteen-chunk task;
3. the route remains stable through C8, C16, both Probe-withdrawal phases, and
   all four zero-Probe checkpoints;
4. the L2-alone boundary near 90% and the approximately seven-point L3 support
   contribution are stable structural properties, not late withdrawal damage.

The registered result is positive, but its scope is one selected successful
training trajectory. It does not estimate how often new seeds acquire this
route, whether L3-dominant seeds form at the same curriculum boundary, or
whether IST exceeds a standard Transformer.

## Registered stop boundary and next study

Level 7.4 is closed as
`l2_core_l3_support_established_by_16chunk_and_stable`. Do not add checkpoints,
conditions, or samples to this completed panel.

The most informative next study is a separately preregistered deterministic
replay of the C2-to-C4 interval with denser frozen milestones. The replay must
start from the original C2 optimizer and RNG state, keep training computation
unchanged, and reproduce the original C4 model hash before its intermediate
checkpoints are interpreted. That would localize the abrupt formation event
inside the currently unresolved 1,000-step interval without changing this
result.
