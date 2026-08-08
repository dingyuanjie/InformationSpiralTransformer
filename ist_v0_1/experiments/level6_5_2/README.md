# Level 6.5.2: cross-stream stability confirmation

Level 6.5.1 isolated the 16-chunk update magnitude and found that `5e-5`
preserved the transferred solution while `1e-4` degraded it on one data stream.
Level 6.5.2 tests whether that difference is robust across five deterministic
training/evaluation streams.

## Fixed factors

- Model checkpoint: `hard400_seed313/stage3.pt`
- Restored Adam optimizer moments
- 16 chunks x 128 tokens = 2,048 total tokens
- Frozen probe and zero probe loss
- Strict deterministic mathematical SDP backend
- 1,000 training steps plus 500 continued maintenance steps

## Compared factor

- `5e-5`: stabilization candidate
- `1e-4`: destructive-update control

Five data-stream seeds are used for each learning rate. The model checkpoint is
held fixed, so this tests robustness to subsequent data streams, not robustness
across independently formed model initializations.

Run locally:

```powershell
python run_level6_5_2_local.py
```

The default uses fewer intermediate evaluation batches than Level 6.5.1 to
reduce runtime; the final evaluation remains 50 batches. Completed LR/stream
pairs are reused unless `--force` is supplied. Results are stored under
`experiments/level6_5_2/formal/`.
