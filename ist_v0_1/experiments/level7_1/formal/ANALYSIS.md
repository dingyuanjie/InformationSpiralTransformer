# Level 7.1 formal analysis

## Decision

- Run integrity: **PASS**.
- Seed1217 formation: **FAIL**, final fresh 16-chunk query accuracy 93.75%
  versus the registered 95% threshold.
- Seed1429 formation: **FAIL**, final fresh 16-chunk query accuracy 90.625%
  versus the registered 95% threshold.
- Seven-condition causal gate: **NOT OPENED**, because neither initialization
  passed formation.
- Secondary frozen read-gap diagnostic: **NOT OPENED**, for the same registered
  reason.
- Registered primary classification: `formation_replication_failed`.
- Strong or conditional independent replication: **NOT SUPPORTED**.

The run completed normally. This is a formal negative result, not a crash,
missing output, or incomplete resume.

## Integrity and protocol execution

The formal run used exactly the two preregistered new model seeds, 1217 and
1429. It did not use seed909, an old checkpoint, an old Level 6.19 evaluation
split, an output-head rescue, a router/factorized repair, a replacement seed,
or an extended training budget. The runner and static preregistration hashes
recorded at completion were:

- runner SHA-256:
  `a500dd0fe5822071bbcc1338141fb77add79297cb655df8aaa1e4da15f78d6f1`;
- static preregistration SHA-256:
  `4e530732ae518cc1db406fd8442fb4765ef5e6fff3d97bf66a48a6680ed0d9b2`.

The aggregate integrity result is **PASS**. The field
`all_formed_seed_integrity_passed=true` is vacuously true because zero seeds
entered the frozen diagnostic stage; it must not be interpreted as evidence
that causal or Probe fingerprints were tested in this run.

## Independent formation

Both initializations learned every curriculum stage within the locked budget.
Both nevertheless failed the separate 800-example final formation evaluation
after the complete auxiliary-Probe withdrawal schedule.

| Seed | Fixed stage | 2 / 4 / 8 / 16-chunk steps | 16-chunk curriculum query | Final query | Shortfall | Final local | Final Probe min | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1217 | PASS at 1,600 | 2,400 / 1,200 / 300 / 400 | 95.00% | **93.75%** | 1.25 pp | 100.00% | 95.875% | 22.75 min |
| 1429 | PASS at 1,600 | 2,500 / 800 / 100 / 200 | 96.25% | **90.625%** | 4.375 pp | 99.375% | 92.375% | 19.66 min |

The combined runtime was approximately 42.41 minutes. All four curriculum
stages reported `passed=true` for both seeds. The negative result therefore
does not mean that the models never acquired the task. It means that the
registered formation-to-maintenance pipeline did not retain >=95% fresh
behavior through the complete withdrawal schedule in either new
initialization.

The short local task remained essentially solved (100.00% and 99.375%), so the
failure is specific to long-range cross-chunk behavior rather than token
recognition in the first chunk.

## Withdrawal dynamics

The small online evaluations during withdrawal were volatile. Seed1217 ranged
from 90.00% to 97.50% query accuracy; seed1429 ranged from 87.50% to 98.75%.
Transient points above 95% cannot be selected post hoc. The registered decision
uses the distinct 800-example final evaluation after all 750 zero-Probe
maintenance updates.

Seed1217 is especially informative descriptively: its final training Probe
minimum remained 95.875% while deployed query behavior was 93.75%. This is
consistent with a representation/readout or maintenance-alignment vulnerability,
but it is not a fresh read-gap confirmation because the preregistered frozen
Probe panel was never opened. Seed1429 ended lower on both measures (92.375%
Probe minimum and 90.625% query).

## Causal replication

The causal panel was conditionally registered only for a seed that first
passed formation. Since no seed did, the following conditions were not run:

- intact;
- reset all Memory;
- zero all Memory;
- batch-roll all Memory;
- zero final-layer Memory;
- batch-roll final-layer Memory;
- keep only final-layer Memory.

Consequently, Level 7.1 does **not** show that persistent Memory causality
failed in a formed model. Its formal classification is formation failure, not
`causal_replication_failed`. Earlier causal results remain evidence about
models that successfully formed, but their unconditional cross-initialization
generality is not established by this stage.

## Frozen read-gap diagnostic

Fresh Memory-concat and query-hidden probes were also conditional on formation
and were not fit. The recorded counts of powered and replicated read-gap seeds
are both zero because the diagnostic was unopened, not because two powered
tests returned null effects.

No conclusion about generalization of the Level 6.19 read-access obstruction
can be drawn from Level 7.1. The final training-Probe values above are secondary
descriptions and cannot replace the registered fresh train/validation/test
Probe protocol.

## Scientific conclusion

Level 7.1 falsifies the strong claim that the current Level 6.8
formation-to-maintenance recipe reliably produces >=95% 16-chunk behavior
across untouched initializations. Both new seeds temporarily crossed their
curriculum gates, yet neither retained the threshold on the larger final split
after Probe withdrawal.

This sharpens the overall IST conclusion:

1. successful checkpoints provide substantial evidence for persistent,
   distributed, causally used Memory;
2. those mechanisms are conditional on successful formation;
3. the present training protocol does not form and stabilize that behavior
   reliably across initialization;
4. auxiliary-Probe withdrawal/zero-Probe maintenance is a principal observed
   boundary, rather than a solved implementation detail;
5. decodability, causal use, behavioral readout, and formation reliability must
   remain separate claims.

The result does not erase the earlier positive causal interventions. It limits
their scope: IST has demonstrated the mechanism in selected successfully
formed models, but has not demonstrated a robust training recipe that produces
the mechanism at the registered behavioral level in new initializations.

## Registered stop boundary

Do not extend either seed, replace it with a third seed, select a transient
withdrawal checkpoint, rescue an output head, change a threshold, or reopen the
Level 6.19 router-repair branch. Any future work on formation reliability must
be a separately preregistered research phase with a new training hypothesis
and new untouched seeds; Level 7.1 is closed as a negative replication.
