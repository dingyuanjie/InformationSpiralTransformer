# Level 6.18.4 formal analysis

## Decision

The preregistered classification is **mixed or ambiguous residual**. The exact
threshold for a Memory-to-query routing diagnosis required a 5-point gap; the
observed all-Memory versus query-hidden gap is 4.59 points.

The quantitative pattern nevertheless rules out Memory loss and strongly
localizes the dominant remaining 16-chunk deficit between persistent Memory
and the final query-token representation. The transferred output head is close
to the best linear readout available from that query token.

This is a frozen post-hoc diagnosis on seed 707. It is not cross-initialization
evidence and does not alter the Level 6.18.1 result.

## Checkpoint compatibility

The untouched and Level 6.18.3 rescued checkpoints differ only in:

- `output.weight`;
- `output.bias`.

The maximum change to every non-output tensor, the original diagnostic Probe,
and output rows 16–18 is exactly zero. Both heads therefore read the same frozen
IST backbone and Memory trajectories.

## Frozen 16-chunk behavior

On the held-out 1,024-example test split:

| Deployed readout | Accuracy |
|---|---:|
| Original task head | 84.47% |
| Transferred 12-chunk rescue head | **92.09%** |
| Original diagnostic Probe | 85.16% |
| Original-head local control | 98.83% |
| Transferred-head local control | 97.36% |

The transferred head corrects 80 original-head errors and harms only 2,
improving accuracy by 7.62 points with 95% CI `[+5.96, +9.28]` and McNemar
`p=1.41e-21`.

## Frozen tomography

| Refitted linear feature | Held-out accuracy |
|---|---:|
| Third-layer Memory mean | 93.55% |
| Third-layer all-slot concatenation | 97.36% |
| All-layer Memory means | 96.68% |
| All-layer/all-slot Memory | **97.56%** |
| Final query-token hidden | **92.97%** |
| Third-layer Memory + query hidden | 97.36% |
| All Memory + query hidden | 97.36% |

The long-range identity is still strongly present in Memory at 16 chunks. All
Memory is 13.09 points above the original head and 5.47 points above the
transferred head. Third-layer all-slot decoding trails all Memory by only 0.20
points, again locating almost all useful information in the third layer.

Adding query hidden to Memory does not improve the Memory decoder. The final
query token therefore contributes no important complementary target signal
that is absent from Memory.

The original Probe's 85.16% is stale: refitting on the same frozen Memory raises
accuracy to 97.56%. This is representation drift rather than Memory loss.

## Where the residual sits

The transferred head and refitted query-hidden decoder are close:

- transferred head: 92.09%;
- query-hidden decoder: 92.97%;
- difference: +0.88 points, CI `[-0.10, +1.95]`, McNemar `p=0.136`.

Thus another output-head replacement on the same query hidden is not supported
as the dominant fix.

In contrast, all-Memory decoding exceeds query-hidden decoding by 4.59 points,
CI `[+3.12, +6.05]`, with 54 corrected versus 7 harmed and McNemar
`p=4.32e-10`. The effect is statistically clear even though its point estimate
falls 0.41 points short of the preregistered 5-point categorical threshold.

Per-sample evidence agrees:

- the transferred head makes 81 errors;
- all-Memory decoding corrects 62 of them (76.54%);
- query-hidden decoding corrects only 19 (23.46%);
- transferred-head/all-Memory oracle union reaches 98.14%;
- query-hidden/all-Memory oracle union reaches 98.24%.

The formal label remains mixed/ambiguous, but the mechanistic weight of evidence
favors partial Memory-to-query-token routing degradation plus a small residual
readout mismatch.

## Shared-trajectory Memory causality

On a separate 1,024-example fixed dataset:

| Condition | Original head | Transferred head | Original local | Transferred local |
|---|---:|---:|---:|---:|
| Intact | 86.13% | **92.97%** | 98.83% | 97.56% |
| Reset | 5.76% | 5.66% | 98.83% | 97.56% |
| Zero | 5.66% | 5.76% | 98.83% | 97.56% |
| Batch-roll | 4.69% | 4.59% | 98.83% | 97.56% |

Both heads pass the causal gate. The transferred-head intact-to-disrupted drop
is 87.21 points. The 16-chunk signal remains sample-specific persistent Memory,
not a local shortcut.

## Scientific conclusion

The 16-chunk residual separates into a clear hierarchy:

1. Memory storage is strong: 97.56% linearly decodable;
2. the final query token exposes only 92.97%;
3. the transferred output head reads 92.09%, statistically indistinguishable
   from the best linear query-hidden decoder.

Level 6.18.3 largely solved the shared output-head mismatch. The next limiting
interface is the transfer of already-stored Memory information into the final
query token at 16 chunks.

## Next falsification test

Level 6.18.5 should perform a surgical routing rescue rather than another
output-head search:

1. retain the successful Level 6.18.3 head;
2. keep embeddings, attention, Memory encoders/updaters, and lower layers
   frozen;
3. update only the final block's Memory read path (`memory_read` and
   `memory_fusion_gate`), whose output affects the final token but not the
   returned persistent Memory state;
4. train on 16-chunk query targets with protected 8- and 12-chunk validation;
5. require 16-chunk test accuracy at least 95%, 8/12 retention at least 95%,
   unchanged frozen Memory states, and reset/zero/batch-roll causal passes;
6. compare against a no-update control and the fixed Level 6.18.3 head.

If this restricted intervention closes the 16-chunk gap without changing
Memory states, the routing interpretation is causally confirmed. Seed 909
should remain unopened until the routing protocol is frozen.

