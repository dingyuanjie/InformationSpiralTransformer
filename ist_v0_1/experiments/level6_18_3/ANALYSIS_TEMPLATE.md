# Level 6.18.3 formal analysis

Status: not run.

## Head fitting and validation selection

- Best fitted-head epoch:
- Fully fitted 12-chunk validation accuracy:
- Selected interpolation alpha:
- Selected 8-chunk validation accuracy:
- Selected 12-chunk validation accuracy:
- Selection gate:

## Protected paired tests

| Chunks | Untouched | Head-only rescue | Change and 95% CI |
|---:|---:|---:|---:|
| 8 | | | |
| 12 | | | |
| 16 | | | |

- 12-chunk corrected / harmed:
- 12-chunk McNemar result:
- 16-chunk zero-shot interpretation:

## Mutation audit

- Changed tensors:
- Maximum non-output change:
- Unused output rows 16–18 change:
- Audit gate:

## Rescued Memory causality

- Intact query:
- Reset query:
- Zero query:
- Batch-roll query:
- Minimum local accuracy:
- Causal gate:

## Decision

- Formal Level 6.18.3 pass:
- Does this causally confirm output-head alignment?:
- What remains initialization- or length-specific?:
- Next falsification test:

Use `formal/result.json` as the numerical source of truth. A successful
head-only rescue is a mechanism intervention on seed 707, not evidence that the
original Level 6.18.1 formation gate passed.

