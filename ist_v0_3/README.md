# Information Spiral Transformer v0.3

IST v0.3 is a new architecture branch created after v0.2 failed its locked
natural-language cross-chunk generalization gate.

The central change is that Memory no longer compresses a complete chunk into
learned anonymous slots.  It retains a small set of actual source-layer token
states together with exact provenance:

- input token id;
- absolute stream position;
- source chunk id;
- selection score;
- query-time retrieval weight.

The Reader retrieves a sparse query-dependent top-k set. `zero`, cross-example
`swap`, and slot `shuffle` interventions are built into the architecture.  The
pretrained adapter writes and reads at the same frozen decoder-layer boundary and
keeps the no-history path identical to the backbone.

Milestone 0 implements the architecture and causal invariants. Milestone 1 is a
source-span coverage gate; it is a retention diagnostic, not yet a language-
generalization claim.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_smoke.py
python -m pytest tests -q
```

Run the first gate before authorizing any long training run:

```powershell
python run_v0_3_coverage_gate.py --dry-run
python run_v0_3_coverage_gate.py --smoke-test
python run_v0_3_coverage_gate.py --local-files-only
```

The formal gate measures whether at least one answer-span token remains in
Memory after 2/4/8/16 chunks. Every distance must reach 80% span-hit rate. A
failure means selection or retention must be redesigned; training the Reader is
not allowed to hide a failed Writer.
