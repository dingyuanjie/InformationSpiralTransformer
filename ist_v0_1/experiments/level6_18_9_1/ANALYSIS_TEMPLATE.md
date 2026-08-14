# Level 6.18.9.1 formal analysis

## Decision

- classification:
- panel A passed:
- panel B passed:
- protected tests opened: no
- registered next boundary:

## Checkpoint and Memory integrity

Confirm exactly four changed final `memory_read` tensors, 16,640 parameters,
unchanged original Probe, and exact persistent-Memory invariance.

## Panel A

| Chunks | Source accuracy | Candidate accuracy | Accuracy change CI | Margin change CI | Cross-entropy change CI | Pass |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | | | | | | |
| 12 | | | | | | |
| 16 | | | | | | |

## Panel B

| Chunks | Source accuracy | Candidate accuracy | Accuracy change CI | Margin change CI | Cross-entropy change CI | Pass |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | | | | | | |
| 12 | | | | | | |
| 16 | | | | | | |

## Cross-panel agreement

State whether 8/12 retention and 16-chunk absolute accuracy plus margin
superiority replicate independently. Do not average away a failed panel.

## Scientific conclusion

Distinguish validation-calibration evidence from protected-test evidence.

## Next experiment

Follow only `decision.registered_next_boundary` in `formal/result.json`.
