# IST v0.5.1 result

Status: `complete`

## Oracle Reader stability

| Condition | Chunks | Seeds | Accuracy | Std |
|---|---:|---:|---:|---:|
| oracle_current | 2 | 5 | 0.9250 | 0.0678 |
| oracle_current | 4 | 5 | 0.9141 | 0.0674 |
| oracle_current | 8 | 5 | 0.8703 | 0.0753 |
| oracle_current | 16 | 5 | 0.8266 | 0.0433 |
| oracle_current | 32 | 5 | 0.7969 | 0.0459 |
| oracle_stable | 2 | 5 | 1.0000 | 0.0000 |
| oracle_stable | 4 | 5 | 1.0000 | 0.0000 |
| oracle_stable | 8 | 5 | 1.0000 | 0.0000 |
| oracle_stable | 16 | 5 | 0.9547 | 0.0140 |
| oracle_stable | 32 | 5 | 0.9000 | 0.0392 |

## 32-chunk capacity audit

The stream contains 64 fact occurrences. `K/N` is the exact-occurrence ceiling for an unbiased query-blind reservoir.

| Condition | K | K/N | Exact retained | Same binding | Accuracy | Acc given retained | Acc given absent |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle_current | 4 | 0.0625 | 0.0609 | 0.1609 | 0.1828 | 0.9023 | 0.1382 |
| oracle_current | 8 | 0.1250 | 0.1187 | 0.2469 | 0.2469 | 0.8408 | 0.1709 |
| oracle_current | 12 | 0.1875 | 0.1812 | 0.3187 | 0.2938 | 0.7942 | 0.1856 |
| oracle_current | 16 | 0.2500 | 0.2500 | 0.4062 | 0.3484 | 0.7840 | 0.2077 |
| oracle_current | 24 | 0.3750 | 0.3875 | 0.5484 | 0.4125 | 0.7385 | 0.2074 |
| oracle_current | 32 | 0.5000 | 0.5188 | 0.6984 | 0.4813 | 0.6755 | 0.2718 |
| oracle_current | 64 | 1.0000 | 1.0000 | 1.0000 | 0.5469 | 0.5469 | 0.0000 |
| oracle_stable | 4 | 0.0625 | 0.0563 | 0.1422 | 0.1781 | 1.0000 | 0.1293 |
| oracle_stable | 8 | 0.1250 | 0.1219 | 0.2547 | 0.2625 | 0.9292 | 0.1707 |
| oracle_stable | 12 | 0.1875 | 0.1766 | 0.3406 | 0.3250 | 0.8904 | 0.2059 |
| oracle_stable | 16 | 0.2500 | 0.2437 | 0.4141 | 0.3844 | 0.8680 | 0.2301 |
| oracle_stable | 24 | 0.3750 | 0.3937 | 0.5422 | 0.4469 | 0.7831 | 0.2289 |
| oracle_stable | 32 | 0.5000 | 0.5141 | 0.6547 | 0.5188 | 0.7602 | 0.2575 |
| oracle_stable | 64 | 1.0000 | 1.0000 | 1.0000 | 0.6062 | 0.6062 | 0.0000 |

Oracle rows force the supervised exact target into Memory and are diagnostics, not deployable scores.
Binding interventions are valid only when original accuracy is above chance.
