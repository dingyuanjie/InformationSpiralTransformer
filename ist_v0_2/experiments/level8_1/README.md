# Level 8.1 — Old Information Retention

The first Chunk contains one marked target; subsequent stream Chunks are noise. v0.1 and hierarchical v0.2 receive identical matched training at 2/4/8/16 Chunks and identical evaluation streams at 1/2/4/8/16/32/64/128/256/512/1000 Chunks.

```powershell
python run_level8_1_local.py --smoke-test
python run_level8_1_local.py
```

The formal endpoint is the rightmost milestone whose pooled Wilson 95% lower bound exceeds `1/16`, plus paired per-example McNemar comparisons. A longer lifetime is a hypothesis, not an assumed outcome.
