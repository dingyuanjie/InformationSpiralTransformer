# v0.4 Milestone 2.2.2: relational Query alignment

This is a fresh Query/Key training run on 24-token events with stride 8. Every
event containing the complete entity token span and answer token is a valid
positive. Qwen, Writer, Value/Output and lifecycle remain frozen. Training uses
12 entities, two query templates and 128 answer tokens; validation locks out
four entities, one query template and 32 answer tokens at 4/8/16 chunks.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_relational_query_alignment.py --dry-run
python run_v0_4_relational_query_alignment.py --local-files-only
```

Do not load or resume the old eight-token Query checkpoint. Use `--resume` only
for checkpoints created in this new output folder.
