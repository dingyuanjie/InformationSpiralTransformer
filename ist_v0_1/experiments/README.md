# Reproducible experiments

Run commands from `ist_v0_1` with Python 3 and PyTorch 2.x.

## Results archive

`results/` contains the raw JSON and visualization produced during v0.1-v0.3.
Files are committed unchanged so reported numbers remain auditable.

## Level 4

```bash
python level4_long_context.py --encoding rope --seed 313
```

Each run creates `level4/<encoding>_seed<seed>/` containing `metrics.json` and
stage checkpoints. Re-run with `absolute`, `sinusoidal`, or `dynamic_rope` for
the position-encoding comparison. CUDA is selected automatically when present.

Resume the verified 512-token checkpoint and extend the curriculum to 1024 and
2048 tokens:

```bash
python level4_long_context.py --encoding rope --seed 313 \
  --resume-checkpoint experiments/level4/rope_seed313/checkpoint_512_509.pt
```

The extended stages use batch sizes 8 and 4 respectively to bound GPU memory.
