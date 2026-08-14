# Level 6.19 formal analysis

## Decision

**Level 6.19 passes its frozen diagnostic protocol and classifies the remaining
16-chunk errors as `hard_example_memory_to_read_access_failure`.** The formally
passed Level 6.18.3 source model makes 277 errors on 4,096 new diagnostic
examples (93.24% accuracy). An independent linear decoder applied to all final
persistent-Memory slots recovers the correct label on 227/277 = **81.95%** of
those errors.

The first registered material loss is from persistent Memory to read context:
decoder accuracy on source errors falls from 81.95% to 40.07%, a **41.88-point
drop**. This exceeds the preregistered 15-point threshold and has priority over
later losses. The registered next boundary is therefore:

> Read routing/slot selection is the first hard-example obstruction.

This is a localization result, not a rescued model. No trained deployment
checkpoint is produced by this level.

## Integrity and matching

All registered integrity conditions passed:

- the source is the formally passed Level 6.18.3 seed707 checkpoint at 16
  chunks;
- the failed Level 6.18.9 candidate was not loaded;
- seed909 and the protected tests remained unopened;
- all model parameters were frozen and no model update occurred;
- the manually reconstructed final query path agrees exactly with the native
  forward path (maximum absolute error `0.0`);
- probe-train (2,048), probe-validation (512), and diagnostic (4,096) seeds are
  disjoint;
- the 277 source errors exceed the registered minimum of 200.

Every source error was matched without replacement to one source-correct case
using nearest top1-minus-top2 deployed confidence. Both groups contain 277
examples. Mean confidence is 1.3008 for errors and 1.3454 for matched correct
examples; mean absolute pair distance is 0.0450 and the maximum is 0.2188.
Thus the control panel is close in deployed confidence, although confidence
matching cannot make the groups identical in all latent difficulty factors.

The probes were trained only on the independent train split and selected on
the independent validation split. Diagnostic labels were used for evaluation,
grouping, and the registered correct-versus-rival directions, not for fitting
the probes.

## Interface decoding

| Interface | Overall | Source errors | Matched correct | Error-minus-matched, 95% CI |
|---|---:|---:|---:|---:|
| persistent Memory | 98.36% | 227/277 (81.95%) | 270/277 (97.47%) | -15.52 pp [-20.58, -10.47] |
| pre-fusion feature | 90.77% | 84/277 (30.32%) | 210/277 (75.81%) | -45.49 pp [-52.71, -37.91] |
| read context | 92.26% | 111/277 (40.07%) | 218/277 (78.70%) | -38.63 pp [-45.85, -31.05] |
| fusion delta | 87.84% | 66/277 (23.83%) | 188/277 (67.87%) | -44.04 pp [-51.26, -36.82] |
| fused feature | 92.77% | 109/277 (39.35%) | 224/277 (80.87%) | -41.52 pp [-48.38, -34.30] |
| FFN output side branch | 76.05% | 81/277 (29.24%) | 157/277 (56.68%) | -27.44 pp [-35.03, -19.13] |
| pre-norm residual | 92.60% | 59/277 (21.30%) | 219/277 (79.06%) | -57.76 pp [-64.26, -50.90] |
| query hidden | 93.38% | 60/277 (21.66%) | 238/277 (85.92%) | -64.26 pp [-70.76, -57.40] |
| deployed logits, independent probe | 91.38% | 50/277 (18.05%) | 203/277 (73.29%) | -55.23 pp [-62.09, -48.01] |

All paired accuracy differences in the table have sign-flip `p=0.00010`.
These tests show that equally confident source errors carry less readily
decodable task information after the Memory interface than matched correct
examples.

The persistent-Memory result is especially strong. Across all 4,096 cases, the
source behavior and the independent Memory decoder have the following joint
outcomes:

| Source behavior | Memory decoder correct | Memory decoder wrong |
|---|---:|---:|
| correct | 3,802 | 17 |
| wrong | 227 | 50 |

The source accuracy is 3,819/4,096 = 93.24%, while the Memory decoder reaches
4,029/4,096 = 98.36%. An oracle that accepts either the source decision or the
Memory decoder reaches 4,046/4,096 = **98.78%**. This oracle is diagnostic and
not a deployable result, but it rules out absent linearly decodable Memory
information as the dominant explanation for the source errors.

### Sample-level transitions on the 277 source errors

| Adjacent independent decoders | correct/correct | correct/wrong | wrong/correct | wrong/wrong | Net change |
|---|---:|---:|---:|---:|---:|
| Memory -> read context | 109 | 118 | 2 | 48 | -116 |
| read context -> fused feature | 84 | 27 | 25 | 141 | -2 |
| fused feature -> pre-norm residual | 45 | 64 | 14 | 154 | -50 |
| pre-norm residual -> query hidden | 45 | 14 | 15 | 203 | +1 |
| query hidden -> deployed-logit probe | 36 | 24 | 14 | 203 | -10 |

Among only the 227 errors whose persistent Memory is decoded correctly, the
read-context decoder remains correct on 109/227 = 48.02%, the fused decoder on
104/227 = 45.81%, the pre-norm residual decoder on 57/227 = 25.11%, and the
query-hidden decoder on 59/227 = 25.99%. The first transition therefore loses
118 Memory-decodable cases and recovers only two Memory-undecodable cases. A
second registered loss occurs from fused feature to pre-norm residual (18.05
points), but it is downstream of the larger, first-priority read loss.

The isolated FFN output remains a side diagnostic. It is not the residual
stream and is not used to order the causal boundary.

## Gradient accessibility

| Metric | Source errors | Matched correct | Paired error-minus-matched, 95% CI | p |
|---|---:|---:|---:|---:|
| Memory gradient norm | 0.07364 | 0.08493 | -0.01129 [-0.03231, +0.00884] | 0.287 |
| context gradient norm | 0.20870 | 0.21411 | -0.00541 [-0.03418, +0.02327] | 0.716 |
| fused gradient norm | 0.56817 | 0.54125 | +0.02691 [-0.02360, +0.07831] | 0.304 |
| pre-norm residual gradient norm | 2.12863 | 2.00251 | +0.12613 [+0.01806, +0.23359] | 0.0215 |
| deployed-gradient/Memory-code cosine | -0.01542 | -0.01738 | +0.00196 [-0.00518, +0.00896] | 0.601 |
| directional access to Memory code | -0.00400 | -0.00192 | -0.00208 [-0.01281, +0.00779] | 0.704 |
| attention mass on top-four code slots | 11.89% | 12.73% | -0.837 pp [-1.613, -0.088] | 0.0335 |
| gradient fraction on top-four code slots | 13.02% | 13.75% | -0.731 pp [-1.488, +0.012] | 0.0585 |

There is no evidence of a simple vanishing-gradient failure at Memory, context,
or fused feature: their group differences are small and their intervals cross
zero. Residual gradients are slightly larger, not smaller, on errors. The
deployed gradient is nearly orthogonal to the independent Memory-code direction
in both groups, so its group difference also does not explain errors by itself.

The selective-targeting measure is more informative. For each case, the four
slots contributing most strongly to the independent correct-versus-deployed-
rival Memory code receive less read attention on errors than on confidence-
matched correct cases. Their gradient fraction is lower in the same direction,
although its interval narrowly includes zero. This supports a targeting
mismatch rather than a lack of gradient magnitude.

## Slot composition and targeting

The strongest error-group decoder contributions are concentrated in slots 2,
11, 0, 7, and 17:

| Slot | Mean Memory-code contribution | Mean read attention | Mean gradient norm |
|---:|---:|---:|---:|
| 2 | 0.3324 | 3.23% | 0.01342 |
| 11 | 0.3296 | 3.01% | 0.01438 |
| 0 | 0.3200 | 2.70% | 0.01372 |
| 7 | 0.2994 | 3.44% | 0.01448 |
| 17 | 0.2810 | 2.72% | 0.01342 |

By contrast, the most-attended slots on errors are 16, 18, 25, 20, and 24,
whose Memory-code contributions are only 0.1141, 0.1277, 0.1130, 0.1188, and
0.1123. Their attention ranges from 4.49% to 4.91%. Across the 32 aggregate
slot rows, the post-hoc Pearson correlation between code contribution and
attention is -0.19; code contribution and gradient norm correlate +0.45, while
attention and gradient norm correlate +0.65.

The aggregate rankings and correlations are descriptive because they combine
cases with different correct labels and deployed rivals. The preregistered
per-example top-four attention result is the stronger evidence: hard cases
systematically under-attend their own highest-contribution slots relative to
matched correct cases.

## Scientific conclusion

Level 6.19 distinguishes **information absence** from **information access**.
Persistent Memory contains linearly accessible correct-label information for
most deployed errors: 81.95% are decoded correctly at Memory, and a diagnostic
source-or-Memory oracle reaches 98.78%. Most of that advantage is lost on the
very first read into a 64-dimensional context.

The result is not consistent with a simple explanation that hard cases have no
useful Memory code or no gradient reaching Memory. It is consistent with the
frozen query read allocating insufficient attention and gradient mass to the
slots that carry the relevant code. The later fused-to-residual loss shows that
read routing is not necessarily the only obstruction, but the preregistered
ordering makes it the first boundary to test.

Independent probe accuracy is an information-localization measurement. A lower
probe accuracy after an interface does not by itself prove literal information
destruction, and a probe can exploit directions that the deployed model never
uses. Likewise, attention and gradient association alone are not a causal
intervention. The warranted claim is therefore:

> Correct-label information is usually present in persistent Memory on the
> remaining hard cases, but the first reproducible obstruction is access to
> that information through the deployed Memory-to-context read.

This does not rehabilitate the failed Level 6.18.9 optimizer branch or establish
a new 95% checkpoint.

## Next experiment

The next milestone should be **Level 6.19.1: selective slot-read causal
intervention**, not another optimizer search.

Keep the Level 6.18.3 model frozen, seed909 and protected tests locked, and use
new disjoint 16-chunk panels. On source errors stratified by whether the
independent Memory probe is correct, intervene only on the Memory-to-context
read:

1. increase or preserve-normalize read mass on each example's preregistered
   top-k Memory-code slots, with fixed `k` and dose levels;
2. compare against equal-dose random-slot, low-contribution-slot,
   attention-shuffle, and no-op controls;
3. measure context correct-versus-rival margin and decoder accuracy first, then
   actual frozen deployed correction rate;
4. report regressions on confidence-matched correct cases and the full panel;
5. include a context-level positive-control intervention to test whether the
   downstream residual path can use a corrected context at all.

If code-slot targeting selectively improves read context and actual decisions,
the read-routing obstruction becomes causal. If context improves but behavior
does not, the fused-to-residual loss becomes the next boundary. If targeted
slot interventions do not improve context beyond controls, the linear
Memory-code ranking is descriptive rather than sufficient and the next study
must test nonlinear/compositional reads instead.
