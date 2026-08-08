# Level 6.14.3: slot-to-slot causal propagation graph

This stage resolves the final-layer pollution pathway at individual destination
slot resolution. After swapping a preregistered source pair and allowing one
chunk of propagation, each of the other 30 slots is restored individually.

For every directed source-pair → destination edge the experiment records:

- paired accuracy recovery;
- reduction in donor-target attraction;
- destination representation displacement and donor projection;
- update-gate strength;
- effective source-to-destination routing.

The effective routing matrix composes the memory read and write operations:
token-to-source `memory_attention` followed by destination-to-token
`compression_weights`. This structural graph is compared with the behavioral
causal graph using Pearson and Spearman correlations.

The formal protocol uses swap-after-4 and swap-after-8, three preregistered
source pairs per model, and 1,200 paired examples per condition. Individual
destination tests receive Holm correction. Source-pair restoration and
all-final-layer restoration are included as controls.

```powershell
python run_level6_14_3_local.py
```

Results are stored under `experiments/level6_14_3/formal/` and resume after
every completed condition.
