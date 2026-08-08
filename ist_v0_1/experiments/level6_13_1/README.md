# Level 6.13.1: targeted pair confirmation

This confirmatory stage freezes candidates selected from the Level 6.13
discovery set, then evaluates them on five new random evaluation seeds.

For each model, the preregistration includes:

- the largest discovered keep-pair gain;
- the largest discovered joint-deletion interaction;
- the two strongest cross-seed robust pairs;
- a performance-matched near-zero interaction control.

Every unique condition stores per-example correctness for 2,000 new examples.
The analysis uses paired McNemar tests, fixed-seed 5,000-iteration bootstrap
confidence intervals, and separate Holm corrections for sufficiency-gain and
pair-necessity test families. Candidate registration is written before any new
confirmation evaluation starts and is reused on resume.

```powershell
python run_level6_13_1_local.py
```

Results are stored under `experiments/level6_13_1/formal/`. The run is
condition-level resumable through each seed's `predictions.json`.
