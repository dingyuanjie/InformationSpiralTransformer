# Level 6.19.5: frozen router observability-supervision diagnosis

Level 6.19.4 localized an eight-head signed correction mechanism but its fixed
label-free signed router recovered only 23.31% of the Oracle margin gain and
did not beat an equal-parameter residual reader. This diagnostic level asks
whether dose/gating, signed direction, their joint coupling, or the projected
memory-value basis is the limiting factor.

## Frozen boundary

- Level 6.18.3 seed707 trunk and all existing probes remain frozen.
- All three Level 6.19.4 routers remain frozen.
- The failed Level 6.18.9 candidate is excluded.
- Seed909 and protected tests remain locked.
- Architecture, model, and optimizer search remain closed.
- Hybrid Oracle arms are label-aware diagnostics, not deployable readers.

## Registered experiment

Four hybrid arms replay the frozen signed reader with learned or Oracle dose
crossed with learned or Oracle direction. Four fixed small probes use exactly
the Level 6.19.4 router observables:

- primary/error-state classifier;
- Oracle-dose regressor;
- signed memory-value direction distiller;
- equal-parameter, matched-trainable-initialization fixed-residual direction
  control.

Probe architecture, initialization, optimizer, and epoch count are fixed.
Validation is reported but never selects an architecture or checkpoint; the
final epoch is always frozen before the one-shot formal diagnostic is opened.

## Formal run

From `ist_v0_1`:

```powershell
python run_level6_19_5_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **8-25 minutes**. The
formal split sizes are 4,096 probe-train, 1,024 validation, and 4,096 fresh
diagnostic examples. Progress is written continuously to
`experiments/level6_19_5/formal/progress.json`.

Run the implementation-only smoke test with isolated seeds and output:

```powershell
python run_level6_19_5_local.py --smoke-test --force
```

Smoke results are not scientific evidence.

## Interpretation

- Oracle dose passes 25% with learned direction, but Oracle direction does not
  with learned dose: dose/gating is dominant.
- Oracle direction passes, but Oracle dose does not: signed direction is
  dominant.
- Both pass while the frozen combination fails: joint coupling is dominant.
- Neither passes: both frozen components are limiting; held-out Probe ceilings
  distinguish poor observability from poor joint supervision.
- Residual direction matching/exceeding signed direction leaves the
  memory-value-basis-specific deployment claim unsupported.

This level is diagnostic only and cannot open seed909 or protected tests.

## Formal artifacts

The formal folder will contain:

- `preregistration.json`;
- `probe_training.json` and `diagnostic_probes.pt`;
- `result.json`, `summary.json`, and `predictions.json`;
- `router_observability_diagnosis.png`;
- `ANALYSIS.md`, completed after the formal run.
