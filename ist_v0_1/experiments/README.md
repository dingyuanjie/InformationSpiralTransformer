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
