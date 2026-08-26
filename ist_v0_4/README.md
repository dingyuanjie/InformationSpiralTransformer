# Information Spiral Transformer v0.4

IST v0.4 replaces isolated-token salience with an explicit memory lifecycle.
It is inspired by functional distinctions in human memory, not a claim to
biologically reproduce the brain.

## What gets remembered

- Every recent event span first enters **working memory** intact.
- Surprising and novel event spans are admitted to **episodic memory**.
- A remembered item is a complete source span with token ids and positions, not
  an anonymous vector or one isolated salient token.
- Repeatedly retrieved episodes are rehearsed into **semantic prototypes**.

## What gets forgotten

- Working memory forgets by recency under a strict capacity.
- Episodic traces compete by strength, age, idle time and retrieval count.
- Rehearsal raises strength and delays eviction.
- Redundant events receive lower novelty and are less likely to consume scarce
  episodic capacity.
- Semantic memory retains consolidated regularities, while exact provenance
  remains an episodic responsibility.

## Milestone 0

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_smoke.py
python -m pytest tests -q
```

This milestone validates lifecycle and provenance invariants only. The next gate
must audit arbitrary-token span coverage before connecting a pretrained model.

## Milestone 1

```powershell
python run_v0_4_lifecycle_gate.py --dry-run
python run_v0_4_lifecycle_gate.py
```

This compares incidental, distinctive, repeated and retrieval-reinforced event
traces across 16/32/64 chunks.

## Milestone 2

```powershell
python run_v0_4_pretrained_writer_gate.py --dry-run
python run_v0_4_pretrained_writer_gate.py --local-files-only
```

This connects a frozen Qwen 0.5B and audits open-token event coverage before any
Reader training.
