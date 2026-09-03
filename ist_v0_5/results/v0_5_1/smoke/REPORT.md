# IST v0.5.1 result

Status: `complete`

## Oracle Reader stability

| Condition | Chunks | Seeds | Accuracy | Std |
|---|---:|---:|---:|---:|
| oracle_current | 2 | 1 | 0.0000 | 0.0000 |
| oracle_current | 4 | 1 | 0.1250 | 0.0000 |
| oracle_stable | 2 | 1 | 0.0000 | 0.0000 |
| oracle_stable | 4 | 1 | 0.1250 | 0.0000 |

## 32-chunk capacity audit

The stream contains 64 fact occurrences. `K/N` is the exact-occurrence ceiling for an unbiased query-blind reservoir.

| Condition | K | K/N | Exact retained | Same binding | Accuracy | Acc given retained | Acc given absent |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle_current | 4 | 0.0625 | 0.1250 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |
| oracle_current | 12 | 0.1875 | 0.1250 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |
| oracle_stable | 4 | 0.0625 | 0.1250 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |
| oracle_stable | 12 | 0.1875 | 0.1250 | 0.1250 | 0.0000 | 0.0000 | 0.0000 |

Oracle rows force the supervised exact target into Memory and are diagnostics, not deployable scores.
Binding interventions are valid only when original accuracy is above chance.
