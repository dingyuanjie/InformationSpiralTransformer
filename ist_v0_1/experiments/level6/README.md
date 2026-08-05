# Level 6 cross-chunk persistent memory

The target appears in chunk 1 and the query in the final chunk. Each attention
window remains fixed at 512 tokens. Compare a stateless Transformer, IST-C with
memory carried across chunks, and IST-C with memory reset every chunk.

Start with one diagnostic run:

```powershell
python run_level6_local.py --variants ist-persistent --seeds 313
```

Then run the full two-seed comparison:

```powershell
python run_level6_local.py
```

The curriculum uses 2, 4, 8 and 16 chunks (1024 through 8192 total tokens).
Results are stored under `experiments/level6/formal/`.
