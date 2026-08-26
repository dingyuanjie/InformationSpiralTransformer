# v0.4 Milestone 2.2.1: relation-complete events

This gate compares non-overlapping 8-token, non-overlapping 16-token,
overlapping 16-token/8-stride and overlapping 24-token/8-stride segmentation on randomly offset natural-language
facts. Success requires the complete entity token span and answer token to occur
inside at least one event in 95% of examples.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_4
python run_v0_4_relation_coverage_gate.py --dry-run
python run_v0_4_relation_coverage_gate.py --local-files-only
```
