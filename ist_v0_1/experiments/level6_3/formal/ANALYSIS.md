# Level 6.3 probe-supervision withdrawal analysis

Starting from the passed 16-chunk Level 6.2 checkpoint, probe-loss weight was
reduced through 0.2 and 0.1, then set to zero. During the final 500-step stage,
probe parameters were frozen and used only as a read-only measurement.

After 500 zero-probe steps, the last scheduled validation was 96.25% query and
96.25% minimum probe accuracy. A separate final 80-sample evaluation produced:

| Metric | Accuracy |
|---|---:|
| Final query | 100% |
| Local target | 100% |
| Probe after chunk 1 | 100% |
| Probe after chunk 16 | 100% |
| Minimum probe across all chunks | 100% |

First-to-final memory cosine similarity was 0.566. Layer update-gate means were
0.463, 0.426 and 0.293. Memory continued to evolve while retaining a linearly
decodable target representation.

## Supported conclusion

Once established by supervised curriculum, the per-layer Spiral Memory behavior
survived 500 additional optimization steps without any direct probe/retention
loss. The final query objective, local objective and diversity regularizer were
sufficient to maintain the cross-chunk representation.

## Limitations

- This is one seed and a synthetic marked-retrieval task.
- Probe supervision was used to establish the memory before withdrawal.
- The zero-probe stage still used query, local and diversity losses.
- Evaluation used 80 samples; a larger multi-seed confirmation is required.

## Next experiment

Repeat probe withdrawal for five seeds and compare against a model trained from
scratch with probe weight zero. This separates curriculum-created memory from
memory that emerges without direct supervision.
