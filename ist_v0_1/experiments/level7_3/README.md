# Level 7.3: cross-initialization layerwise causal atlas

Level 7.2 replicated the causal necessity of complete persistent Memory in a
new initialization but falsified universal final-layer necessity and
sufficiency. Level 7.3 freezes four successfully behaving models—606, 808,
1001, and 1879—and evaluates their layer routing on one new shared diagnostic
panel.

No model, probe, router, or output head is trained. No checkpoint is selected.
The Level 7.2 protected panel is not reused.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_3_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **25–50 minutes**. The
script saves condition-level progress and resumes automatically after
interruption; do not add `--force` when resuming.

## Atlas

Every model is evaluated on the same 2,048 fresh 16-chunk examples under:

- intact, reset-all, zero-all, and batch-roll-all controls;
- zero and batch-roll interventions for each of the three layers;
- keep-only interventions for each single layer;
- keep-only interventions for each pair of layers.

Layer signatures use locked 20% necessity/misassignment and 90% sufficiency
thresholds. The experiment confirms heterogeneity only if all four models first
pass complete-Memory causality and at least two distinct signatures remain.

## Smoke test

```powershell
python run_level7_3_local.py --smoke-test --force
```

Smoke output is isolated under `experiments/level7_3/smoke/` and is not
scientific evidence.
