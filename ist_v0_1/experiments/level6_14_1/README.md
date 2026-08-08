# Level 6.14.1: pollution recovery dynamics

This stage tests why a one-time cross-example memory swap remains harmful while
zeroing the same slots is usually recoverable.

Three preregistered slot pairs per model are swapped after chunks 1, 4, 8, and
12. Pollution is either left untouched or remediated after 0, 1, 2, or 4
additional chunks by:

- zeroing the polluted slots;
- restoring those slots from a parallel clean counterfactual trajectory.

The analysis records ordinary accuracy and the fraction of predictions that
match the donor example's target when donor and recipient targets differ. This
directly distinguishes generic disruption from donor-identity transfer.

The formal protocol uses three new evaluation seeds, 1,200 paired examples per
condition, paired McNemar tests, Holm correction within each remediation
family, and fixed-seed bootstrap confidence intervals. A remediation scheduled
at or after the final query is retained as a timing negative control.

```powershell
python run_level6_14_1_local.py
```

Results are written to `experiments/level6_14_1/formal/` and resume after every
completed condition.
