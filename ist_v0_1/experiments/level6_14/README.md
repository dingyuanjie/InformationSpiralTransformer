# Level 6.14: dynamic causal trajectories

This stage traces confirmed final-layer memory-slot pairs over chunk time. It
uses candidates frozen by Level 6.13.1 and intervenes after chunks 1, 2, 4, 8,
12, and 15 with four complementary operations:

- `zero_once`: erase the pair once and allow subsequent recovery;
- `zero_persistent`: erase the pair after every later chunk;
- `swap_once`: exchange the pair across batch items once, preserving its
  marginal distribution while breaking example identity;
- `keep_pair_persistent`: remove every other final-layer slot from that time
  onward, measuring when the selected pair becomes sufficient.

The formal protocol uses three new evaluation seeds and 1,200 paired examples
per condition. Per-example predictions are saved. Every pair × intervention
family receives paired McNemar tests with Holm correction across time and
fixed-seed bootstrap confidence intervals.

```powershell
python run_level6_14_local.py
```

Results are stored under `experiments/level6_14/formal/`. Each completed
condition is saved immediately and the run resumes automatically.
