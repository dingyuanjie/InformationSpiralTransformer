# Level 6.18.6 formal analysis

## Decision

Fill from `formal/summary.json`:

- classification:
- first numerically changed interface:
- source largest 16-chunk decodability drop:
- update-500 largest 16-chunk decodability drop:
- registered next boundary:

## Checkpoint boundary audit

Report whether exactly six tensors and 24,896 parameters changed, whether all
changes were under `blocks.2.memory_read.*` or
`blocks.2.memory_fusion_gate.*`, and whether the original Memory Probe remained
unchanged.

## Held-out behavior

| Chunks | Source query | Update-500 query | Change | 95% CI | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 8 | | | | | |
| 12 | | | | | |
| 16 | | | | | |

## Interface tomography

For each length, report independently refitted held-out accuracy for source and
update 500.

| Interface | 8 source | 8 update | 12 source | 12 update | 16 source | 16 update |
|---|---:|---:|---:|---:|---:|---:|
| persistent Memory | | | | | | |
| pre-fusion feature | | | | | | |
| read context | | | | | | |
| fusion gate | | | | | | |
| fused feature | | | | | | |
| FFN output | | | | | | |
| query hidden | | | | | | |
| refit on deployed logits | | | | | | |
| deployed argmax | | | | | | |

## Same-example representation shift

Confirm that persistent Memory and the pre-fusion feature are exactly invariant.
Then identify the first changed interface and distinguish a numerical change
from a held-out decodability improvement.

## Causal Memory controls

Summarize source versus update-500 intact/reset/zero/batch-roll behavior at 16
chunks. These rows diagnose whether the update altered Memory dependence; they
are not a success gate.

## Scientific conclusion

State only what the frozen same-example comparison establishes. A changed
representation without improved held-out decoding is not a mechanism rescue.
Do not infer seed-909 transfer.

## Next experiment

Use the preregistered classification and `registered_next_boundary` from
`formal/summary.json`. Do not start an optimizer search from this result.
