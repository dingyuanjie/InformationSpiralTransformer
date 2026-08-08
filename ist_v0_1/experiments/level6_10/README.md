# Level 6.10: frozen-state memory tomography

IST checkpoints are frozen. Disjoint synthetic train/validation/test sets are
used only to fit post-hoc decoders to the final per-layer, per-slot memory state
at 2, 4, 8, and 16 chunks.

Compared decoders:

- original trained mean-pooled linear probe (no refit);
- refitted mean-concatenated linear probe;
- one linear probe per layer mean;
- one linear probe per individual layer/slot (96 probes trained in parallel);
- one linear probe per layer with all 32 slots concatenated;
- one linear probe with all layers and slots concatenated;
- a 256-hidden-unit nonlinear MLP over all layers and slots.

Features are standardized using training statistics. Probe selection uses only
the validation split; final accuracy uses a disjoint test split. IST receives
no gradients.

```powershell
python run_level6_10_local.py
```

Results are stored under `experiments/level6_10/formal/` and completed
seed/length pairs are reused on restart.
