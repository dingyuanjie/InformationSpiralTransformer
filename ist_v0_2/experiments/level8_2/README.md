# Level 8.2 — Important vs Noise (protocol registered, not yet a result)

Train with two input types: `important` items are queried after a delay; `noise` items are never queried. Record per Chunk and per layer `p_fast`, `p_slow`, `p_episodic`, and `p_forget`, together with write rates, retention, slot age/usage/replacement, and target similarity.

Primary hypothesis: queried important items receive more Slow/Episodic routing than matched noise. Required causal confirmation: zero/freeze Slow and Episodic separately after training. This directory currently registers scope only; it contains no formal result.
