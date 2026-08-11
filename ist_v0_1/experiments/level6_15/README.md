# Level 6.15: inference-time pollution-robust interventions

This stage targets the identified horizontal copying path inside final-layer
memory updates without retraining the checkpoints.

Three intervention families are evaluated:

- fixed propagation attenuation over the full trajectory;
- one-step attenuation only during the first update after pollution;
- an adaptive gate based on cosine consistency between current encoded tokens
  and content read from old memory.

Every defense is evaluated on clean examples and on preregistered slot-pair
swaps after chunks 4 and 8. Metrics include clean accuracy loss, polluted
accuracy, donor-target attraction, paired recovery confidence intervals, and
the remaining clean-to-polluted gap. The script reports candidate Pareto points
that lose no more than one percentage point of clean accuracy.

```powershell
python run_level6_15_local.py
```

Results are written to `experiments/level6_15/formal/` and resume after every
completed condition.
