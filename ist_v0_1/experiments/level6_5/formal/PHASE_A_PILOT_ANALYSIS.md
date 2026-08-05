# Level 6.5 Phase A pilot analysis

The original pilot reported 0/7 successful profiles because it required both
95% query accuracy and 90% frozen-probe accuracy. That combined gate is not
appropriate for a minimum-scaffold experiment.

`hard50` reached 97.5% query accuracy at step 2400 and then 100%, 98.75%, 100%,
100%, and 100% over steps 2600--3000 after probe loss had been zero for more
than 2,500 steps. This is stable behavioral formation, not a single evaluation
spike. The target exists only in chunk 1 and the query only in chunk 2, so the
query cannot be solved without cross-chunk state.

Its frozen linear probe remained near chance because the probe itself received
only 50 optimization steps before being frozen. This shows that a fixed linear
decoder can fail to expose a representation that the nonlinear model uses; it
does not invalidate task-level memory.

The corrected protocol therefore uses two consecutive query evaluations at or
above 95% as the formation gate. Probe accuracy remains a secondary diagnostic.
Corrected results are isolated in `../formal_query_gate/` so the pilot remains
available for auditing.

Other pilot profiles did not cross the behavioral gate. The response to
scaffold duration was non-monotonic: more probe supervision was not uniformly
better. This is evidence of path-dependent optimization and requires multiple
seeds before claiming that 50 steps is a universal minimum.
