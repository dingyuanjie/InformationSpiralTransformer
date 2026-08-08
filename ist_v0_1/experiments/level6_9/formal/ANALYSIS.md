# Level 6.9 analysis: causal persistent-memory intervention

## Registered result

Three behaviorally successful Level 6.8 models were evaluated without any
training under intact, reset, zeroed, and batch-identity-shuffled memory. Each
model was tested at 2, 4, 8, and 16 chunks with 400 examples per condition.

- Model-level causal passes: **3/3**.
- Model x context-length causal passes: **12/12**.
- Mean query drop under the strongest intervened result: **91.52 percentage
  points**.
- Minimum query drop over all model/length combinations: **89.25 points**.

## Mean accuracy across three models

| Chunks | Intact query | Reset query | Zero query | Batch-roll query | Local, all conditions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 99.42% | 7.08% | 7.58% | 4.83% | 99.58% |
| 4 | 99.25% | 5.92% | 5.92% | 6.00% | 99.33% |
| 8 | 99.33% | 6.67% | 7.00% | 7.08% | 99.75% |
| 16 | 96.25% | 5.83% | 6.00% | 5.92% | 98.92% |

The 16-class chance level is 6.25%. Every intervention leaves query accuracy at
or near chance while intact memory remains at 95.5--100% for every individual
model and length. Local accuracy remains 97.75--100% in all conditions.

## Causal interpretation

The reset and zero conditions show that persistent state continuity is
necessary. The batch-roll condition is stronger: it preserves the presence,
shape, scale, and distribution of real model-generated memory tensors, but
assigns each sample another sample's memory. Its chance-level result shows that
generic memory activation is insufficient; the state must carry the correct
sample-specific content.

The unchanged local accuracy rules out general model damage or corrupted first-
chunk processing as the explanation for query collapse. The manipulated factor
is the cross-chunk state pathway.

Together, the interventions establish that the long-range query behavior is
causally dependent on persistent memory continuity, nonzero content, and sample
identity. It is not explained by the query marker, token-position shortcuts, or
ordinary within-chunk competence.

## Probe discrepancy resolved

At 16 chunks, intact mean query is 96.25% while the mean-pooled linear probe is
only 41.25%; seeds 808 and 1001 have especially low probe accuracy. Yet all
memory interventions collapse their query to chance. Therefore these models
do use persistent memory even when the chosen linear probe cannot decode it.

At 2--8 chunks the same probe is approximately 99% accurate, then loses
decodability at 16 chunks while behavioral decoding remains strong. The useful
code changes with repeated memory updates: it may become slot-specific,
distributed across layers, or nonlinearly readable. Mean-pooling all slots and
layers destroys information that the model's learned read path retains.

## Conclusion

This is the strongest evidence produced by the Level 6 series. The experiments
now demonstrate not only correlation and performance, but a causal mechanism:
IST can carry sample-specific information through 16 separate 128-token chunks,
and disrupting that state removes the 2,048-token capability while sparing
local computation.

## Recommended next experiment

Stop optimizer tuning. The next useful study is frozen-state memory tomography:
collect per-layer, per-slot states at 2/4/8/16 chunks and fit post-hoc probes
without updating the IST model. Compare mean-pooled linear, slot-wise linear,
concatenated linear, and a small nonlinear probe. This will locate where the
16-chunk code lives and explain why the original diagnostic fails while the
model succeeds.

