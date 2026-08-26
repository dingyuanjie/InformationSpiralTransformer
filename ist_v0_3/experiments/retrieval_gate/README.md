# v0.3 Milestone 2: query retrieval and causal read

This frozen, untrained diagnostic separates three questions: whether Query
top-k touches the retained fact span, whether cross-example Memory swap removes
that self-fact alignment, and whether Memory changes the correct-answer logit.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_retrieval_gate.py --dry-run
python run_v0_3_retrieval_gate.py --local-files-only
```

Results are written to `results.json`; complete terminal output and tracebacks go
to `run.log`. Failure at this untrained gate authorizes Reader alignment
training, but does not invalidate the Milestone 1 Writer result.
