# Frozen Memory 0.4.2 — Full-Context Teacher

Frozen Qwen reads the complete 1K example and supplies final hidden-state plus full-vocabulary logits. The student reads two 512-token chunks and can transmit information only through Fast persistence. Default effective batch is four via micro-batch two and accumulation two.

```powershell
python run_pretrained_frozen_memory_0_4_2.py --dry-run
python run_pretrained_frozen_memory_0_4_2.py --local-files-only
```
