# Level 6.18.4 formal analysis

Status: not run.

## Checkpoint compatibility

- Changed tensors between checkpoints:
- Maximum non-output change:
- Probe change:
- Audit gate:

## 16-chunk deployed behavior

- Original head:
- Transferred 12-chunk rescue head:
- Original diagnostic Probe:
- Local controls:

## Refitted tomography

- Third-layer mean:
- Third-layer all slots:
- All Memory:
- Query hidden:
- Third-layer Memory + query:
- All Memory + query:

## Per-sample residual

- Original errors corrected by transferred head:
- Transferred-head errors corrected by query Probe:
- Transferred-head errors corrected by Memory Probe:
- Oracle union accuracies:

## Shared-trajectory causality

- Original intact / strongest disrupted / local:
- Transferred intact / strongest disrupted / local:
- Causal gates:

## Frozen diagnosis

- Classification:
- Memory minus transferred head:
- Query hidden minus transferred head:
- Memory minus query hidden:
- Next falsification test:

Use `formal/result.json` as the numerical source of truth. This is a mechanism
diagnosis on seed 707, not cross-initialization evidence.

