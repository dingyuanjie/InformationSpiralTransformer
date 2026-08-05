# Level 6.4 stability and spontaneous-memory controls

`maintenance` reuses the same passed 16-chunk checkpoint but trains on five new
data-stream seeds for 500 zero-probe steps. This measures maintenance robustness,
not initialization robustness.

`scratch-zero` uses five independent random model initializations and never uses
probe loss. It applies the 2/4/8/16-chunk curriculum and tests whether persistent
memory emerges from query and local losses alone.

Start with one scratch run:

```powershell
python run_level6_4_local.py --modes scratch-zero --seeds 313
```

Run the full experiment:

```powershell
python run_level6_4_local.py
```
