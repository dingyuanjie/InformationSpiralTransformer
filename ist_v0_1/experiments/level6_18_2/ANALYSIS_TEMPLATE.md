# Level 6.18.2 result analysis

Status: not run.

## Frozen checkpoint behavior

- 8-chunk control query:
- 12-chunk task-head query:
- 12-chunk original probe:
- 12-chunk local control:

## Refitted held-out probes

- third-layer mean:
- third-layer all-slot concatenation:
- all-Memory concatenation:
- query-token hidden:
- third-layer Memory + query hidden:
- all Memory + query hidden:

## Per-sample decoupling

- Memory probe correct / task head wrong:
- Query-hidden probe correct / task head wrong:
- Oracle union:

## Causal control

- intact query:
- reset query:
- zero query:
- batch-roll query:
- minimum local accuracy:
- causal gate:

## Frozen diagnosis

- Classification:
- Memory minus task-head gap:
- Query-hidden minus task-head gap:
- Memory minus query-hidden gap:
- Interpretation:

Use `formal/result.json` as the numerical source of truth. Level 6.18.2 is a
diagnostic result and must not be reported as retroactively passing Level
6.18.1.

