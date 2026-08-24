# NL-1 Formal Analysis

## Integrity

- Complete: Transformer, IST v0.1, and IST v0.2; five seeds each (15 runs).
- Evaluated validation, held-out, and OOD at 2K, 4K, 8K, 16K, and 32K.
- 160 examples per architecture/distance/split; all stored shortcut audits passed.

## Result

This is a valid **negative bridge result**. Eight-way chance accuracy is 12.5%. Held-out normal accuracy stayed near chance:

| distance | Transformer | IST v0.1 | IST v0.2 |
|---:|---:|---:|---:|
| 2K | 10.63% | 13.13% | 13.13% |
| 4K | 12.50% | 9.38% | 9.38% |
| 8K | 13.75% | 10.00% | 10.00% |
| 16K | 14.37% | 17.50% | 17.50% |
| 32K | 10.63% | 13.75% | 13.75% |

Every 95% Wilson interval includes chance. The isolated 16K peak is not monotonic, is not stable across seeds, and is therefore not evidence of long-distance retention.

## Causal controls

IST v0.2 normal was almost unchanged by roll or by independently zeroing Fast, Slow, or Episodic Memory. Reset/zero-all changed some predictions but did not reveal above-chance behavior. Consequently no causal natural-language Memory claim is supported.

## Diagnosis boundary

The experiment does not yet prove that synthetic-to-language Memory transfer is impossible. The 64-hidden models were trained from scratch on UTF-8 bytes, and the protocol did not first establish same-chunk or zero-delay semantic task learnability. Training loss and one-sample stage accuracy were also unstable. Thus language grounding/task acquisition and long-term retention are confounded.

## Decision

Gates A, B, and D fail; do not enter 0.5B modification yet and do not proceed to NL-2. Run NL-1.1 learnability calibration at same-chunk, 512, 1K, and 2K with a fixed held-out evaluation. Only after normal accuracy is clearly above chance at short distance should distance-dependent Memory conclusions be drawn.
