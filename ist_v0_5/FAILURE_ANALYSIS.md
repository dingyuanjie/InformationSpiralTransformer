# Failure classification protocol

When Level A fails, diagnose in this order and change only one cause at a time:

1. `split_audit.passed == false`: data leakage/protocol bug.
2. train accuracy remains near chance for every model: task or optimizer is not learnable.
3. `writer_relation_recall` is low: candidate selection/capacity failure.
4. Writer recall is high but `reader_relation_hit` is low: Query–Evidence representation failure.
5. Reader hit is high but answer accuracy is low: fusion/output failure.
6. Short distance passes but long distance fails: retention/capacity/age competition failure.
7. Normal succeeds but zero/swap does not reduce accuracy: shortcut or ineffective causal intervention.
8. Hybrid does not beat Evidence-only: Core adds no demonstrated value and should be removed or redesigned.
9. Only one seed succeeds: optimization instability, not a stage success.

Do not proceed to Qwen, LoRA, 1B models or extra auxiliary losses until the relevant Level A failure is isolated.
