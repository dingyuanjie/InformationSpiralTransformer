# v0.4 Milestone 2: frozen-Qwen Writer lifecycle

The Qwen 0.5B backbone is frozen. Exact eight-token source events are captured
at one decoder layer and tested with open answer tokens under single exposure,
spaced repetition and oracle retrieval rehearsal at 4/16/32 chunks. No Reader
or language output module is trained.

All sixteen events in a 128-token chunk are initially eligible for episodic
encoding. Selection happens through later lifecycle competition rather than a
write-time guess about which four events will matter.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_pretrained_writer_gate.py --dry-run
python run_v0_4_pretrained_writer_gate.py --local-files-only
```

Results and tracebacks are mirrored to `results.json` and `run.log`.
