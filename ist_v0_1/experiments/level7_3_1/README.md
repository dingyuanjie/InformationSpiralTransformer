# Level 7.3.1: high-precision seed1879 L2-route replication

Level 7.3 established cross-initialization layer-routing heterogeneity, but
seed1879's keep-L2 result was close to the registered 90% sufficiency boundary:
90.9180% with a 95% Wilson interval of [89.5957%, 92.0871%]. Level 7.3.1
freezes that checkpoint and repeats the L2/L3 causal contrast on a larger,
entirely new panel.

No model, Probe, output head, or router is trained. The fixed sample size is
8,192 and may not be extended after seeing the result.

## Formal run

From `ist_v0_1`:

```powershell
python run_level7_3_1_local.py
```

Expected RTX 5060 Laptop GPU runtime is approximately **20-40 minutes**. The
runner saves every completed condition and resumes automatically after an
interruption. Do not add `--force` when resuming.

## Primary decision

The strongest classification requires all registered 95% Wilson gates to
pass. In particular:

- keep-L2 query lower bound must be at least 90%;
- zero-L2 and batch-roll-L2 query upper bounds must be at most 20%;
- zero-L3 and batch-roll-L3 query lower bounds must be at least 80%;
- keep-L3 query upper bound must be at most 20%.

The positive pair control keeps L2 and L3 and requires a query lower bound of
at least 90%. Complete-Memory disruptions and local behavior are checked
again on the same new panel.

## Smoke test

```powershell
python run_level7_3_1_local.py --smoke-test --force
```

Smoke output is isolated under `experiments/level7_3_1/smoke/` and is not
scientific evidence.
