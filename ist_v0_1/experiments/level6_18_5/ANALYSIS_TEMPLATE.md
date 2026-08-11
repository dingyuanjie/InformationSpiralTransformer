# Level 6.18.5 formal analysis

Status: not run.

## Stable routing formation

- Stable update:
- Confirmation 8 / 12 / 16:
- Consecutive confirmations:
- Training gate:

## Protected paired tests

| Chunks | Level 6.18.3 baseline | Routing rescue | Change and 95% CI |
|---:|---:|---:|---:|
| 8 | | | |
| 12 | | | |
| 16 | | | |

- 16-chunk corrected / harmed:
- 16-chunk McNemar result:

## Parameter and Memory audit

- Changed tensors:
- Maximum disallowed parameter change:
- Maximum returned-Memory difference:
- Audit gates:

## Rescued causality

- Intact query:
- Reset query:
- Zero query:
- Batch-roll query:
- Minimum local accuracy:
- Causal gate:

## Decision

- Formal pass:
- Is Memory-to-token routing causally confirmed?:
- Remaining limitation:
- Cross-initialization next step:

Use `formal/result.json` as the numerical source of truth. A successful routing
rescue is still a mechanism intervention on seed 707, not a replacement for the
failed Level 6.18.1 formation result.

