# NL-1.1 Short-range Learnability

This calibration separates language/task acquisition from persistence. It uses four shuffled choices and tests 512, 1K, and 2K tokens across validation, held-out, and OOD.

```powershell
python run_nl_1_1_local.py --dry-run
python run_nl_1_1_local.py --smoke-test
python run_nl_1_1_local.py
```
