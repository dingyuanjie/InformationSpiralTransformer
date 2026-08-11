# Level 6.18.7 formal analysis

## Decision

- classification:
- integrity passed at 8/12/16:
- 16-chunk context effect:
- 16-chunk gate-after-context effect:
- 16-chunk total route effect:
- registered next boundary:

## Checkpoint and transplant integrity

Confirm the six-tensor/24,896-parameter checkpoint boundary, exact persistent
Memory invariance, and exact donor reproduction for context+gate, fused
feature, FFN output, and query hidden in both directions.

## Baseline behavior

| Chunks | Source | Update 500 | Change | 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 8 | | | | | |
| 12 | | | | | |
| 16 | | | | | |

## Forward transplant: update into source

| Chunks | Source | Updated context through source gate | Updated gate activation only | Updated context+gate | Full update |
|---:|---:|---:|---:|---:|---:|
| 8 | | | | | |
| 12 | | | | | |
| 16 | | | | | |

Report the Holm-adjusted 16-chunk primary context, gate-after-context, and total
route tests.

## Reverse restoration: source into update

| Chunks | Update | Source context through updated gate | Source gate activation only | Source context+gate | Source |
|---:|---:|---:|---:|---:|---:|
| 8 | | | | | |
| 12 | | | | | |
| 16 | | | | | |

Use this as a directional corroboration, not a second discovery family.

## Mechanism conclusion

Distinguish linear decodability from causal deployed usefulness. If updated
context alone does not improve source behavior, the Level 6.18.6 decoding gain
is not aligned with the deployed computation. If it improves behavior but the
full update loses the gain, the gate causally cancels it.

The isolated gate-activation patch combines a gate computed under one context
with the other context and is synthetic; do not interpret it alone as a gate
parameter main effect.

## Next experiment

Follow only `diagnosis.registered_next_boundary` from `formal/result.json`.
Do not broaden training or open seed 909 from an invalid transplant audit.
