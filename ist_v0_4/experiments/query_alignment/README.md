# v0.4 Milestone 2.2: natural-language Query alignment

Only Query, event-Key and QueryNorm are trained. Qwen, Writer, lifecycle,
Value/Output, injection gate and source events remain frozen. Each example has
four entity-token facts, varied questions and random event positions. The loss
selects the complete event containing the requested entity's answer token.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_query_alignment.py --dry-run
python run_v0_4_query_alignment.py --local-files-only
```

Use `--resume` after interruption. Checkpoints are ignored by Git and logs are
saved automatically.
