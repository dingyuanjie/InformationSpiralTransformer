# Reproducible experiments

Run commands from `ist_v0_1` with Python 3 and PyTorch 2.x.

## Level 7 evidence bundle

The Level 6 router-repair branch is closed. Build the frozen evidence audit and
lightweight GitHub/reviewer bundle without training or opening protected seeds:

```bash
python run_level7_0_local.py
```

See `level7_0/EVIDENCE_LEDGER.md` for the eight-claim mechanism ledger and
`level7_0/README.md` for bundle contents and scientific boundaries.

## Level 7.1 independent replication

Train two untouched initializations and independently retest 16-chunk
persistent-Memory formation, final-layer causality, and the frozen read gap:

```bash
python run_level7_1_local.py
```

This protocol uses new seeds 1217 and 1429. It cannot replace a failed seed,
extend its training budget, rescue its output head, or open seed909.

## Level 7.2 retention selection

Test one new formation-reliability hypothesis on untouched seeds 1601 and 1879:

```bash
python run_level7_2_local.py
```

Training is unchanged. A locked 1,024-example validation panel selects at most
one of four registered zero-Probe checkpoints before a one-time 4,096-example
protected test and conditional causal audit.

## Level 7.3 layerwise causal atlas

Freeze successful models 606, 808, 1001, and 1879 and test layer-routing
heterogeneity on one new shared panel:

```bash
python run_level7_3_local.py
```

This is a read-only mechanistic replication: no training, checkpoint selection,
protected-data reuse, output-head repair, or router repair.

The formal run completed with integrity PASS. Complete persistent Memory was
causal in all four checkpoints, while three exact layer signatures and two
dominant routing classes were observed. Seeds 606/808/1001 were L3 dominant;
seed1879 was L2 dominant. The registered classification is
`cross_initialization_layer_heterogeneity_confirmed`. See
`level7_3/formal/ANALYSIS.md` for the complete atlas and scope limits.

## Level 7.3.1 high-precision L2-route replication

Freeze the seed1879 checkpoint and repeat its L2/L3 causal contrast on a new
fixed 8,192-example panel, replacing point-estimate sufficiency with a locked
95% Wilson lower-bound decision:

```bash
python run_level7_3_1_local.py
```

This is a precision replication only. It does not reopen Level 7.3, train a
model, select a checkpoint, or permit optional sample extension.

The formal fixed-N run completed with integrity PASS. Eight of nine confidence
gates passed. Keep-L2 reached 90.3442%, but its 95% Wilson lower bound was
89.6856%, so the registered classification is
`l2_route_supported_but_single_layer_sufficiency_inconclusive`. L2 remained
causally necessary, while L2+L3 was robustly sufficient at 96.7773%. The
result supports an L2-core, L3-supported route rather than an L2-only circuit.
See `level7_3_1/formal/ANALYSIS.md`.

## Level 7.4 frozen route-formation trajectory

Evaluate ten preexisting seed1879 checkpoints from 2-chunk curriculum through
zero-Probe step750 on one new shared 16-chunk causal panel:

```bash
python run_level7_4_local.py
```

The registered trajectory separates unformed early behavior from genuine
route migration and tests whether the L2-core/L3-support circuit was already
established at 16-chunk curriculum completion and remained stable throughout
Probe withdrawal. No checkpoint is trained, selected, added, or omitted.

The formal run completed with integrity PASS and classification
`l2_core_l3_support_established_by_16chunk_and_stable`. The route appeared
earlier than required: C2 was unformed at 20.51% fresh 16-chunk query, while C4
reached 95.80% and already had the final L2-core/L3-support signature. Every
checkpoint from C4 through zero-Probe step750 retained that class, with no
route migration. See `level7_4/formal/ANALYSIS.md`.

## Level 7.4.1 deterministic dense formation replay

Restore the original seed1879 C2 model, Probe, optimizer, and RNG states and
replay the unchanged C4 curriculum stage with frozen model milestones at step
1 and every 100 steps:

```bash
python run_level7_4_1_local.py
```

The fresh causal panel opens only if the replay endpoint exactly matches the
original C4 model, Probe, optimizer, CPU/CUDA RNG, validation history, and stop
step. Qualified milestones then localize the stable L2-core/L3-support onset
within the previously unresolved C2-to-C4 interval.

The formal replay completed with integrity PASS and classification
`exact_replay_and_single_stable_formation_transition`. The exact C4 endpoint
matched the original model, Probe, optimizer, CPU/CUDA RNG, validation history,
and stop step. An L2 causal precursor appeared by step600, while the first
fully qualified 16-chunk L2-core/L3-support milestone was step1000; the locked
formation interval was step900-to-step1000.

## Level 7.5 prospective cross-initialization formation dynamics

Prospectively repeat the dense C2-to-C4 analysis on three untouched model
initializations (`2203`, `2551`, and `2909`):

```bash
python run_level7_5_local.py
```

Each seed starts from scratch and stops after C4. The C2 endpoint, C4 step1,
and every 100-step C4 validation milestone are evaluated on one new shared
N=1,024 16-chunk panel with all sixteen whole-Memory, layerwise, and pairwise
interventions. The registered primary test requires the L2 causal precursor
to precede the full L2-core/L3-support route in at least two of three seeds.
Alternative layer routes are reported separately and cannot be relabeled as
the primary replication. See `level7_5/README.md` for runtime, resume behavior,
and the fixed outcome rules.

The formal run completed with integrity PASS and classification
`alternative_route_formation_observed`. The seed1879 L2 two-stage path did not
replicate; instead, all three new seeds formed whole-Memory-causal 16-chunk
behavior through the same L3-dominant signature. See
`level7_5/formal/ANALYSIS.md`.

## Level 7.5.1 fixed-to-C2 route-bifurcation replay

Restore and exactly replay the fixed-stage-to-C2 interval for the three new
L3-dominant seeds and the exceptional seed1879 L2 trajectory:

```bash
python run_level7_5_1_local.py
```

Each seed's 16-chunk causal trajectory opens only after an exact C2 endpoint
match covering model, Probe, optimizer, CPU/CUDA RNG, validation history, and
stop step. Qualified replays evaluate the fixed endpoint, step1, and every
100-step C2 milestone on a new shared N=1,024 full-layer panel. The primary
question is whether weak L3 selection forms prospectively in all three default
L3 trajectories but remains absent from seed1879. See
`level7_5_1/README.md` for the 4-7 hour runtime and fixed stop boundary.

The formal run completed all 57 milestones with integrity PASS and
classification `default_L3_precursor_divergence_confirmed`. Weak L3 selection
first appeared at step700 in seeds2551/2909 and step1000 in seed2203, then
persisted through C2; seed1879 never entered that registered route through
step2300. A post-hoc mirrored diagnostic additionally found a transient weak
L2 scaffold only in seed1879 at steps1400 and1600. That secondary observation
is explicitly exploratory and motivates an independent frozen-panel
confirmation in Level 7.5.2. See `level7_5_1/formal/ANALYSIS.md`.

## Level 7.5.2 independent weak-L2 precursor confirmation

Evaluate the frozen C2 route window on a completely new causal dataset:

```bash
python run_level7_5_2_local.py
```

The formal protocol fixes seed1879 steps1200-1800, with steps1400 and1600 as
the two registered weak-L2 positives. Nine checkpoints bracketing the three
default seeds' L3 transitions are frozen as L2-negative controls. Every one of
the 16 checkpoints receives all sixteen 16-chunk interventions at N=4,096 on
dataset seed7520000. No training or checkpoint selection occurs. The primary
confirmation requires 2/2 seed1879 positives and 0/9 default-route L2 false
positives. See `level7_5_2/README.md` for the fixed protocol, runtime, and
resume behavior.

The formal run completed with integrity PASS and classification
`weak_L2_precursor_partially_replicated`: step1400 replicated but step1600
failed one frozen preservation clause. The new panel independently selected
steps1300-1400 as the weak-L2 window, yielded zero L2 false positives across
the nine default-route controls, and redetected all five L3 calibration
positives. From step1500 onward L2 remained dominant while L3 removal caused a
greater-than-five-point loss, suggesting recruitment of L3 support rather than
loss of the L2 scaffold. See `level7_5_2/formal/ANALYSIS.md`.

## Level 7.5.3 route-commitment causal intervention

Run fixed-compute training-time counterfactual branches around each seed's
registered C2 commitment window:

```bash
python run_level7_5_3_local.py
```

Each of four seeds receives an exact intact replay, 200 steps of selected-layer
Memory suppression, and a matched other-layer suppression branch. All masks
are released before the remaining C2 and C4 schedule. The twelve fixed C4
endpoints are classified on one new shared N=2,048, 16-chunk, sixteen-condition
panel. The primary test requires an L2-specific effect in seed1879 and the
converse L3-specific effect in at least two of three default seeds. See
`level7_5_3/README.md` for exact gates, runtime, and resume behavior.

The formal run completed with integrity PASS and classification
`transient_suppression_disrupts_routes_nonspecifically`. Selected-layer masks
changed the registered endpoint class in 2/4 seeds, but matched other-layer
masks changed it in 4/4. No branch switched between L2 and L3 topology; all
formal class changes were sub-90% long-context formation, while every branch
still reached 96.25%-100% on four-chunk validation. The result rejects a simple
layer-specific commitment switch and instead implicates distributed training
scaffolding and long-context formation efficiency. See
`level7_5_3/formal/ANALYSIS.md`.

## Level 7.5.3.1 unsuppressed recovery dynamics

Resume all twelve frozen Level 7.5.3 endpoints with no masks and a common
additional 1,000-step C4 budget:

```bash
python run_level7_5_3_1_local.py
```

Five-condition N=1,024 screens at recovery steps0/100/300/600/1000 measure
recovery timing and dominant layer retention. A separate N=2,048 full panel at
step1000 provides the registered final route classification. The primary
question is whether all six previously unformed branches recover their original
L2/L3 route while all six already formed branches remain stable. See
`level7_5_3_1/README.md` for the frozen groups, outcomes, and resume behavior.

The formal run completed with integrity PASS and classification
`continued_C4_destabilizes_preformed_routes`. Three of six initially unformed
branches recovered their original route, but only one of six initially formed
branches retained it at the endpoint, and none remained stable across every
registered milestone. No endpoint migrated to the opposite L2/L3 topology.
The result identifies route formation as a path-dependent, metastable training
state rather than monotonic recovery.

## Level 7.5.3.2 optimizer-state × data-stream causal bifurcation

Fork four frozen, outcome-stratified Level 7.5.3 sources under orthogonal
optimizer-state and RNG/data-stream interventions:

```bash
python run_level7_5_3_2_local.py
```

The exact Level 7.5.3.1 continuation is hash-locked and reevaluated rather than
retrained. Twelve new branches reset AdamW state, reset the stochastic stream,
or reset both while holding source weights, C4 loss, learning rate, and compute
fixed. Shared screens at steps0/300/600/1000 and full step1000 panels determine
whether volatility is controlled primarily by optimizer momentum, data order,
their interaction, or the endpoint weight basin. See
`level7_5_3_2/README.md` for the frozen endpoints, runtime, and stop boundary.

The formal run completed with integrity PASS and classification
`optimizer_and_data_stream_both_causal`. Resetting AdamW state materially
changed 3/4 diagnostic trajectories and their final fate in 3/4. Resetting the
data stream materially changed all 4 trajectories, while changing the final
route in 1/4. Resetting both changed all 4. No intervention produced a stable
recovery across the registered milestones, and no opposite L2/L3 route formed.
See `level7_5_3_2/formal/ANALYSIS.md` for the endpoint table and interpretation
boundary.

## Level 7.5.3.3 Memory parameter-group causal intervention

Hold the exact Level 7.5.3.2 optimizer state and data stream fixed while
freezing layer-specific Memory pathways or update gates:

```bash
python run_level7_5_3_3_local.py
```

The four outcome-stratified endpoints receive exact reference, L2 pathway,
L3 pathway, L2 update-gate, and L3 update-gate branches. The registered panels
at steps0/300/600/1000 test whether route volatility is localized to a layer,
to the update gate, or to distributed Memory parameters. See
`level7_5_3_3/README.md` for the fixed parameter groups and stop boundary.

The formal run completed with integrity PASS and classification
`distributed_l2_l3_memory_pathway`. Freezing the L3 Memory pathway affected all
4 diagnostic endpoints and freezing the L3 update gate produced the same
material-effect status in all 4. Freezing the L2 pathway affected 3/4 and the
L2 gate 3/4, including a persistent-L2 recovery and a catastrophic cross-layer
collapse. The result localizes route volatility to a distributed L2/L3 Memory
pathway, with L3 as the strongest direct control point but no single universal
owner. See `level7_5_3_3/formal/ANALYSIS.md`.

## Results archive

`results/` contains the raw JSON and visualization produced during v0.1-v0.3.
Files are committed unchanged so reported numbers remain auditable.

## Level 4

```bash
python level4_long_context.py --encoding rope --seed 313
```

Each run creates `level4/<encoding>_seed<seed>/` containing `metrics.json` and
stage checkpoints. Re-run with `absolute`, `sinusoidal`, or `dynamic_rope` for
the position-encoding comparison. CUDA is selected automatically when present.

Resume the verified 512-token checkpoint and extend the curriculum to 1024 and
2048 tokens:

```bash
python level4_long_context.py --encoding rope --seed 313 \
  --resume-checkpoint experiments/level4/rope_seed313/checkpoint_512_509.pt
```

The extended stages use batch sizes 8 and 4 respectively to bound GPU memory.

## Level 7.5.3.4

冻结 Memory pathway 的 slot 查询、写入核心、读取/融合三类参数，比较 L2/L3 的细粒度因果贡献。正式结果位于 `experiments/level7_5_3_4/formal/`。

```powershell
python run_level7_5_3_4_local.py --dry-run
python run_level7_5_3_4_local.py --smoke-test --force
python run_level7_5_3_4_local.py
```
