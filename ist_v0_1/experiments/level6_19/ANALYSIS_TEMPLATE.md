# Level 6.19 formal analysis

## Decision

- classification:
- source errors / diagnostic samples:
- persistent-Memory decoder accuracy on source errors:
- first material information drop:
- registered next boundary:

## Integrity and matching

Confirm exact gradient-path reconstruction, fully frozen parameters, disjoint
splits, minimum error count, and confidence matching quality.

## Interface decoding

| Interface | Overall | Source errors | Confidence-matched correct | Error-minus-matched |
|---|---:|---:|---:|---:|
| persistent Memory | | | | |
| pre-fusion feature | | | | |
| read context | | | | |
| fusion delta | | | | |
| fused feature | | | | |
| FFN output side branch | | | | |
| pre-norm residual | | | | |
| query hidden | | | | |
| deployed logits | | | | |

## Gradient accessibility

Compare Memory/context/fused/residual gradient norms, deployed-gradient versus
Memory-code alignment, and correct-rival margin between hard and matched groups.

## Slot composition and targeting

Report the slots with strongest hard-example decoder contribution, read
attention, gradient norm, and any mismatch between stored code and deployed
targeting. Treat slot analyses as descriptive unless preregistered otherwise.

## Scientific conclusion

Distinguish absent Memory information from inaccessible information and avoid
interpreting isolated FFN output as the residual stream.

## Next experiment

Follow only `analysis.diagnosis.registered_next_boundary` in `result.json`.
