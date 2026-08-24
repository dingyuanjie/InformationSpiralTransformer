# Level 8.2 — Important vs Noise

Train with two input types: `important` items are queried after a delay; `noise` items are never queried. Record per Chunk and per layer `p_fast`, `p_slow`, `p_episodic`, and `p_forget`, together with write rates, retention, slot age/usage/replacement, and target similarity.

Primary hypothesis: queried important items receive more Slow/Episodic routing than matched noise. Required causal confirmation: zero/freeze Slow and Episodic separately after training.

The executor starts from the completed Level 8.1 hierarchical checkpoints. `zero_*` is applied only at query time and measures read dependence; `freeze_*` is applied throughout the delay and measures write/maintenance dependence.

```powershell
python run_level8_2_local.py --dry-run
python run_level8_2_local.py --smoke-test
python run_level8_2_local.py
```
