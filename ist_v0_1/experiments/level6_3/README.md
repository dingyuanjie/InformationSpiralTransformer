# Level 6.3 probe supervision withdrawal

Starting from the passed 16-chunk Level 6.2 checkpoint, this experiment keeps
the context at 16 x 128 tokens and lowers probe-loss weight from 0.5 to 0.2,
0.1 and finally 0. The final zero-probe stage runs for 500 steps while the probe
is used only for measurement.

```powershell
python run_level6_3_local.py
```

Pass criteria: final query accuracy >=95% and minimum probe accuracy across all
16 chunks >=90% after the zero-probe stage.
