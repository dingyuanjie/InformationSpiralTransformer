# Level 6.14.4: group restoration dose curves

This stage tests the collective scale of final-layer pollution by first
restoring the polluted source pair, then additionally restoring K = 1, 2, 4,
8, 16, or 30 destination slots after one propagation chunk.

Destination groups are selected by frozen Level 6.14.3 rankings:

- donor-projection strength;
- effective source-routing strength;
- discovery-set single-slot causal effect;
- five fixed random nested rankings.

The experiment uses new evaluation seeds. It measures accuracy recovery,
donor-target attraction reduction, recovery fraction, and deviation from the
sum of discovery single-slot effects. Restoring only the source pair, all 30
destination slots, and all 32 final-layer slots provides decomposition and
positive controls. Group tests use paired bootstrap intervals, McNemar tests,
and Holm correction.

```powershell
python run_level6_14_4_local.py
```

The selection plan is frozen before evaluation in
`experiments/level6_14_4/formal/selection_plan.json`. Runs resume after every
unique restored subset.
