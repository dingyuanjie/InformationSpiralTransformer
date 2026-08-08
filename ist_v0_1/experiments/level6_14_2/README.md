# Level 6.14.2: pollution propagation tomography

This stage localizes where swapped final-layer memory content propagates.
Clean and polluted trajectories run in parallel. After every chunk, the script
records all 3 layers × 32 slots using:

- relative L2 displacement from the clean trajectory;
- projection onto the corresponding donor-example displacement;
- cosine alignment with the donor direction.

For each preregistered pair and swap time, pollution is then repaired after one
or two chunks at three scopes:

- only the original source pair;
- all final-layer memory slots;
- all memory slots in all three layers.

This distinguishes source-slot persistence, within-layer spread, cross-layer
spread, and any residual non-memory trajectory effect. The formal protocol uses
1,200 paired samples per condition, bootstrap confidence intervals, paired
McNemar tests, and Holm correction.

```powershell
python run_level6_14_2_local.py
```

Outputs are stored under `experiments/level6_14_2/formal/`, including per-pair
tomography and restoration-scope plots. Runs resume after each condition.
