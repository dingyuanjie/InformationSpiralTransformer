# Level 6.6: full independent formation validation

This is the faithful independent-initialization test of the originally
successful training path. Every seed starts from random model and probe weights.

1. Level 6.1 fixed-marker two-chunk diagnostic with direct probe supervision.
2. Random-marker 2/4/8/16-chunk curriculum with probe weight 0.5.
3. Stabilized 16-chunk learning rate `5e-5`.
4. Probe withdrawal weights 0.2, 0.1, and 0.0.
5. Final 400-example evaluation after 500 zero-probe steps.

Strict mathematical SDP determinism is enabled. Each seed and completed stage
is checkpointed independently. Completed seeds are reused on restart.

```powershell
python run_level6_6_local.py
```

Results are written under `experiments/level6_6/formal/`.
