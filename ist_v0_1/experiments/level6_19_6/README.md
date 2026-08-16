# Level 6.19.6: one frozen factorized signed read

## Formal status

**Completed — registered FAIL; router-repair branch closed.** The formal run
passed every integrity and full-accuracy safety audit, but recovered 24.4056%
of the signed Oracle margin gain against the registered 25% minimum and passed
only four of six Holm-corrected specificity contrasts. See
`formal/ANALYSIS.md` for the complete analysis and `BRANCH_CLOSURE.md` for the
active boundary. Seed909 remains locked.

Level 6.19.5 showed that label-free error state and Oracle dose are strongly
observable, that a directly supervised signed direction crosses the 25%
recovery boundary, and that the original joint router remains below it. This
level tests exactly one repair: compose the already frozen dose and signed
direction probes without any new training or calibration.

## Frozen boundary

- Level 6.18.3 seed707 trunk and all existing probes remain frozen.
- All Level 6.19.4 routers remain frozen.
- All final-epoch Level 6.19.5 probes remain frozen.
- No training, threshold selection, probability calibration, architecture
  selection, or checkpoint selection is allowed.
- The failed Level 6.18.9 candidate is excluded.
- Seed909 and protected tests remain locked.
- No second repair formula is allowed.

## Single candidate

```text
factorized delta = frozen predicted dose * frozen signed unit direction
```

The candidate uses only the Level 6.19.4 router observables. The frozen
error-state classifier is reported as an audit but is not used as a gate. This
avoids choosing a classifier threshold or calibration after Level 6.19.5.

## Formal run

From `ist_v0_1`:

```powershell
python run_level6_19_6_local.py
```

The script uses one new 4,096-example formal split with seed `6196100` and
writes progress to `experiments/level6_19_6/formal/progress.json`. Expected RTX
5060 Laptop GPU runtime is approximately **3-10 minutes**.

Implementation-only smoke test:

```powershell
python run_level6_19_6_local.py --smoke-test --force
```

Smoke output uses isolated seeds and is not scientific evidence.

## Registered controls

- source;
- frozen Level 6.19.4 signed router;
- equal-dose residual direction;
- shuffled memory observables;
- rolled factorized delta;
- head-permuted signed coefficients;
- full label-aware signed Oracle ceiling.

Only the factorized signed read is a repair candidate. Controls and the Oracle
cannot be selected as alternatives.

## Success gate

The candidate passes only if it:

1. recovers at least 25% of the full Oracle deployed-margin gain on fresh
   Memory-decodable source errors;
2. beats source and all five mechanism controls in paired margin after Holm
   correction;
3. passes the -0.25 percentage-point full-accuracy noninferiority bound;
4. passes every frozen-state, label-free-inference, L2, and split audit.

Pass means repeat across independent probe initializations before seed909.
Failure permanently stops this router-repair branch.

## Formal artifacts

The formal folder will contain:

- `preregistration.json`;
- `result.json`, `summary.json`, and `predictions.json`;
- `factorized_signed_read.png`;
- `ANALYSIS.md`, completed after the formal run.
