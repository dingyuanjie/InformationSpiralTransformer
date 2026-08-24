# Pretrained Base Route

This track opens only after the Natural Language Bridge gates pass. The first target is an openly licensed approximately 0.5B base model, forked from one checkpoint into an untouched baseline and an IST-Memory branch with the same tokenizer, data, token budget, and steps.

Order: `base_smoke` -> `frozen_memory` -> `partial_unfreeze` -> `causal_ablation` -> `long_context`. Do not start at 7B and do not use subjective chat quality as the primary endpoint.
