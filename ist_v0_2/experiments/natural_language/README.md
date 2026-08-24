# IST Natural Language Bridge

The bridge uses natural UTF-8 text, shuffled answer options, random needle positions, split-specific entities/templates, and exact reserved answer-token scoring. It is not evaluated by chat impressions.

Formal order: NL-1 single fact, NL-2 entity binding, NL-3 state update, NL-4 core/detail, NL-5 conflict, NL-6 multihop. Advancement requires a causal Memory effect and held-out stability.

