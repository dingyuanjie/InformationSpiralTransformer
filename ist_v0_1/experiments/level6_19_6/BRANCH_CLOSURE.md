# Level 6.19 router-repair branch closure

Level 6.19.6 completed normally and produced a valid formal result. The single
registered factorized candidate failed the conjunction because Oracle recovery
was 24.4056% (<25%) and only four of six Holm-corrected specificity contrasts
passed. Integrity and full-accuracy noninferiority passed.

The registered failure boundary is now active:

- no second factorized or router-repair composition;
- no classifier gating, threshold tuning, or dose-cap tuning;
- no substitution of the residual control or prior router as a candidate;
- no reopening optimizer/model search in this branch;
- no seed909 or protected-test evaluation;
- no reuse of seed `6196100` to select a new repair.

Allowed follow-up is limited to consolidating, documenting, plotting, and
independently reproducing claims that are already supported. A genuinely new
architecture or training hypothesis must be defined as a new research phase
with new preregistered data boundaries; it is not a continuation of this repair
search.

Authoritative evidence is in `formal/result.json`, `formal/summary.json`, and
`formal/ANALYSIS.md`.
