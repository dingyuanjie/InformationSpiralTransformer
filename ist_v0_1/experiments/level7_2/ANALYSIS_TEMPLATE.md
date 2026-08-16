# Level 7.2 formal analysis

## Decision

- integrity:
- seed1601 selection / protected / causal:
- seed1879 selection / protected / causal:
- registered classification:
- stop boundary:

## Unchanged training trajectory

Report fixed and curriculum gates, phase 1/2 withdrawal, all four saved
zero-Probe candidates, and runtime. Confirm no budget or optimizer change.

## Validation-only selection

Report all four validation query/local estimates and Wilson intervals. Apply
the eligibility and ranking rule exactly. State whether the protected test was
opened.

## Protected behavior

For the single selected eligible checkpoint, report the one-time 4,096-example
query/local result and Wilson interval. Do not compare or select alternatives
on this panel.

## Conditional causal audit

If opened, report all seven paired conditions and apply the registered final-
layer necessity/sufficiency gate. Otherwise state the upstream gate that kept
it closed.

## Scientific conclusion

Distinguish checkpoint-endpoint instability from formation failure. Do not add
candidates, replace seeds, extend training, rerun protected data, or reopen
output-head/router repair.
