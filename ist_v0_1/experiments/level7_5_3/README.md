# Level 7.5.3: route-commitment causal intervention

## Formal status

The formal run completed with integrity **PASS** and classification
`transient_suppression_disrupts_routes_nonspecifically`. Selected-layer
suppression changed the registered route class in 2/4 seeds, while matched
other-layer suppression changed it in 4/4; no seed met the complete directional
criterion. All formal changes were failures to cross the 90% formation gate,
not L2/L3 route switches: the original dominant layer identity remained intact
at all 12 endpoints. All branches recovered 96.25%-100% four-chunk validation,
but intervention endpoints ranged from 67.58%-93.02% at fresh 16 chunks. See
`formal/ANALYSIS.md` for the full distinction between formation strength and
route identity.

Level 7.5.2 found a reproducible seed1879 L2 scaffold followed by recruitment
of L3 support, while three default trajectories independently retained their
L3 route. Level 7.5.3 tests whether those early layer choices causally direct
the final route rather than merely predict it.

For each of four seeds, the script exactly restores the fixed-stage endpoint
and runs three fixed-compute branches:

1. intact deterministic replay;
2. 200 C2 steps suppressing the route-selected layer;
3. the same 200 steps suppressing the matched other layer.

Seed1879 suppresses L2 versus L3 over steps1201-1400. Seeds2203, 2551, and
2909 suppress L3 versus L2 over their registered preformation windows. Every
branch then releases the mask, finishes the original C2 and C4 compute budget,
and receives a new N=2,048 full 16-condition endpoint panel.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_5_3_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **4-7 hours**. Training
resumes at validation boundaries and causal evaluation resumes by condition.
After interruption, execute the same command without `--force`.

The intact branch for each seed runs first. Its model, Probe, optimizer,
CPU/CUDA RNG, validation history, and C2/C4 endpoints must exactly reproduce
the original trajectory. A failed exact gate closes that seed rather than
opening approximate counterfactual interpretation.

## Registered primary outcome

`layer_specific_route_commitment_confirmed` requires:

- seed1879 L2 suppression to change/prevent the original L2-core/L3-support
  route while matched L3 suppression preserves it; and
- selected-layer L3 suppression to change/prevent the original L3 route in at
  least two of seeds2203/2551/2909, while matched L2 suppression preserves it.

Route recovery, nonspecific disruption, partial specificity, and integrity
failure are frozen alternative outcomes. No threshold or training-budget
search is permitted.

## Smoke test

```powershell
python run_level7_5_3_local.py --smoke-test --force
```

Smoke mode validates all frozen hashes, checks that the no-op training path is
bit-exact to the original update, and verifies that only the requested layer is
zeroed and that suppression changes the update. It is not scientific evidence.

## Outputs

Formal outputs are written under `experiments/level7_5_3/formal/`:

- `result.json`: complete training, endpoint, diagnosis, and integrity record
- `summary.json`: compact branch routes and causal profiles
- `progress.json`: resume/completion state
- `route_commitment_counterfactual.png`: endpoint comparison
- `seed*/<branch>/`: resumable training states, exact gates, and causal panels
