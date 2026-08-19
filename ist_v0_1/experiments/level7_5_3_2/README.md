# Level 7.5.3.2: optimizer-state × data-stream causal bifurcation

Level 7.5.3.1 showed that another 1,000 C4 steps can both repair unformed
routes and destroy preformed routes. This experiment asks which inherited
training state controls that volatility.

Four endpoints are frozen to cover four distinct parent outcomes: persistent
L2 loss, L3 recovery, late L3 collapse, and transient L3 collapse followed by
recovery. This outcome-stratified set is a mechanism diagnostic and is not a
prevalence estimate.

For each endpoint, the exact Level 7.5.3.1 continuation is compared with three
new 1,000-step branches:

- reset AdamW state while preserving the source RNG stream;
- preserve AdamW state while resetting the RNG/data stream;
- reset both AdamW state and the RNG/data stream.

Model and Probe weights are exact at the fork. All branches use unchanged C4
training, learning rate, loss, and compute, with no Memory mask. Screens at
steps 0/300/600/1000 and full final panels use new frozen datasets shared by
all arms.

## Formal run

```powershell
python run_level7_5_3_2_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **3-5 hours**. Training
and every evaluation condition are resumable. After interruption, run the same
command without `--force`.

The formal run creates 12 new training arms, four shared source screens, 48
post-source trajectory screens, and 16 final confirmation panels. The exact
reference is never retrained; its hash-locked Level 7.5.3.1 checkpoints are
reevaluated on the new panels.

## Smoke test

```powershell
python run_level7_5_3_2_local.py --smoke-test --force
```

Smoke mode verifies the source hashes, optimizer-reset gate, RNG preservation,
shared reset RNG state, and a one-step update in every new arm. It is not
scientific evidence.

## Outputs

Formal outputs are written under `experiments/level7_5_3_2/formal/`, including
`result.json`, `summary.json`, `progress.json`,
`optimizer_rng_bifurcation.png`, and resumable per-arm artifacts.

The registered stop boundary forbids adding seeds, endpoints, steps, reset
seeds, learning rates, or post-hoc arms to this level.
