# Frozen Memory 0.5.2

Level 0.5.1 showed that ordinary answer supervision plus a swapped-Memory margin
does not teach unique-stream content binding.  This stage adds a teacher target at
the exact decoder-layer boundary where Memory is injected.

For every example, frozen Qwen first reads the complete 1024-token sequence and a
hook captures the final Query token entering decoder layer 20.  The chunked student
reads two 512-token chunks and its post-injection Query representation is trained to
match that target.  A second objective matches the missing-context delta:

`full_context_teacher - chunked_student_before_injection`

Answer CE and the cross-example swap margin remain active.  Evaluation and
checkpoint selection use held-out normal, zero, reset, and swap conditions.

```powershell
cd D:\code\InformationSpiralTransformer\ist_v0_2
python run_pretrained_layer_memory_0_5_2.py --dry-run
python run_pretrained_layer_memory_0_5_2.py --smoke-test --local-files-only --force
python run_pretrained_layer_memory_0_5_2.py --local-files-only
```
