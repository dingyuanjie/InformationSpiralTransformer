# Level 7.5.3.3: Memory parameter-group causal intervention

Level 7.5.3.2 showed that both inherited AdamW state and the stochastic data
stream causally affect route volatility. This level fixes both of those factors
and intervenes only on layer-specific Memory parameters.

Four outcome-stratified endpoints are frozen: persistent L2 loss, unformed L3
recovery, late L3 collapse, and transient L3 collapse/recovery. Each receives
the exact reference continuation plus four new branches:

- freeze the complete L2 Memory pathway;
- freeze the complete L3 Memory pathway;
- freeze only the L2 update gate;
- freeze only the L3 update gate.

The complete Memory pathway includes the selected layer's `memory`,
`memory_read`, and `memory_fusion_gate` parameters. Frozen parameters retain
their source weights and AdamW state but have `requires_grad=False`; all other
parameters receive the exact same C4 updates, optimizer state, and RNG/data
stream as the reference.

## Formal run

```powershell
python run_level7_5_3_3_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **4-6 hours**. Training
and every evaluation condition are resumable. After interruption, run the same
command without `--force`.

The run creates 16 new training arms, four shared source screens, 60 trajectory
screens, and 20 final confirmation panels. The exact reference is not retrained;
its hash-locked Level 7.5.3.1 checkpoints are reevaluated on the new panels.

## Smoke test

```powershell
python run_level7_5_3_3_local.py --smoke-test --force
```

Smoke mode verifies source-state restoration, parameter-name selection, frozen
parameter invariance after one update, and the parent hash locks. It is not
scientific evidence.

## Outputs

Formal outputs are written under `experiments/level7_5_3_3/formal/`, including
`result.json`, `summary.json`, `progress.json`,
`memory_parameter_group_effects.png`, and resumable per-arm artifacts.

The registered stop boundary forbids adding parameter groups, layers, seeds,
steps, optimizer/RNG interventions, or post-hoc gates.
