# Level 7.5.2: independent weak-L2 precursor confirmation

## Formal status

The formal run completed with integrity **PASS** and classification
`weak_L2_precursor_partially_replicated`. The registered step1400 L2 positive
replicated, while step1600 failed exactly one clause because L3 removal caused
a 6.62-point drop, exceeding the frozen five-point allowance. Step1300 passed
as a registered descriptive-window checkpoint, all nine default-L3 controls
remained L2-negative, and all five L3 calibrations were redetected. The result
supports an early L2 scaffold followed by recruitment of L3 support, but it
does not satisfy the preregistered 2-of-2 full-confirmation rule. See
`formal/ANALYSIS.md`.

Level 7.5.1 formally confirmed that all three default trajectories select a
weak L3 route during C2 while seed1879 never enters it. A post-hoc mirrored
analysis then found an apparent weak L2 scaffold at seed1879 steps1400 and
1600. Level 7.5.2 converts that observation into a frozen out-of-sample test.

No model is trained. The script evaluates seven frozen seed1879 checkpoints
from step1200 through step1800 and nine matched default-L3 control checkpoints
on one new shared N=4,096 causal panel. Both candidate-positive checkpoints,
all negative controls, hashes, rule thresholds, conditions, and classification
outcomes are fixed in `preregistration.json`.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_5_2_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **4-6 hours**. There are
16 frozen milestones, each receiving all sixteen 16-chunk causal conditions at
N=4,096. Progress resumes at the condition and milestone level; after an
interruption, run the same command again without `--force`.

## Registered primary outcome

`independent_weak_L2_precursor_confirmed` requires both registered seed1879
candidate checkpoints (steps1400 and1600) to pass the unchanged mirrored weak
L2 rule on the new panel, while all nine checkpoints from seeds2203, 2551, and
2909 remain L2-negative.

The seed1879 step1200-1800 window is reported descriptively. Redetection of the
known L3 precursor at five default-route checkpoints is a registered secondary
panel-sensitivity check, not an extra hidden primary requirement.

## Smoke test

```powershell
python run_level7_5_2_local.py --smoke-test --force
```

Smoke mode validates all sixteen real checkpoint hashes and the parent exact
replay result, then runs the full intervention code on one small panel. It is
an implementation check, not scientific evidence.

## Outputs

Formal outputs are written to `experiments/level7_5_2/formal/`:

- `result.json`: full source audit, metrics, rule checks, diagnosis, integrity
- `summary.json`: compact milestone results and selectivity profiles
- `progress.json`: resumable completion state
- `preregistration.json`: copied frozen protocol
- `independent_weak_L2_confirmation.png`: target/control trajectory plot
