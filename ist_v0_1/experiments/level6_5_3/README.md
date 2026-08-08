# Level 6.5.3: adaptive late-stage stabilization

Level 6.5.2 showed that fixed `5e-5` was much better than `1e-4` on average but
still stream dependent. This experiment preregisters a one-way adaptive policy.

## Policy

1. Start 16-chunk continuation at learning rate `5e-5`.
2. At each 100-step validation, permanently reduce to `1e-5` when the first of
   these conditions occurs:
   - query accuracy reaches 95%;
   - after step 300, query has fallen at least 10 percentage points from a best
     value of at least 85%;
   - step 600 is reached.
3. Never raise the learning rate again.
4. Save the model with the highest intermediate validation query.
5. After 1,500 total probe-free steps, evaluate both the last model and the
   preselected best checkpoint on the same independent 400-example test stream.

The best-checkpoint selection uses only intermediate validation data. The final
test stream is not used to choose a checkpoint.

Run:

```powershell
python run_level6_5_3_local.py
```

Results are written to `experiments/level6_5_3/formal/` and completed streams
are reused unless `--force` is supplied.
