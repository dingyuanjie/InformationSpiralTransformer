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

Current scope is architecture milestone 0: implementation and causal invariants,
not a language-generalization claim.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_smoke.py
python -m pytest tests -q
```

The next experimental gate should first verify that the selected source tokens
actually cover the fact span before any long training run is authorized.
