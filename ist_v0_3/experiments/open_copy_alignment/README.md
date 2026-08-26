# v0.3 Milestone 2.3: open-copy and entity binding

Qwen and Writer remain frozen. The Reader is warm-started from step 400 and
trained on 128 dynamically selected single-token answers, four entity-fact pairs
per example, varied templates and random locations. Losses cover full-vocabulary
next-token prediction, target-slot retrieval and alternating zero/swap margins.
Thirty-two answer tokens are locked out for the next evaluation stage.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_3
python run_v0_3_open_copy_alignment.py --dry-run
python run_v0_3_open_copy_alignment.py --local-files-only
```

Use `--resume` after interruption. Checkpoints are ignored by Git; structured
training results and logs are written beside this README.
