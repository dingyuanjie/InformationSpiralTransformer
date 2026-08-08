# Level 6.13: exhaustive pairwise causal analysis

This stage exhaustively evaluates all `C(32, 2) = 496` final-layer memory-slot
pairs under two complementary interventions:

- `keep-pair`: keep only the selected pair, measuring joint sufficiency;
- `leave-pair-out`: remove only the selected pair, measuring joint necessity.

For every pair the analysis reports:

- keep-pair accuracy;
- gain over the better constituent slot;
- additive keep interaction relative to 16-way chance;
- accuracy drop after deleting both slots;
- joint-deletion interaction beyond the two single-slot drops.

The formal protocol evaluates 992 pair interventions per checkpoint, using the
same 400 examples per condition. All three independent Level 6.8 checkpoints
are included. Progress is saved after every condition and automatically
resumes. Level 6.12 formal results are required for the single-slot baselines.

```powershell
python run_level6_13_local.py
```

Outputs are written to `experiments/level6_13/formal/`, including per-seed and
aggregate JSON summaries and four-panel causal heatmaps.
