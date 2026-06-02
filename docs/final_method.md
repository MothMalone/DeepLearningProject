# Final Method: rewind3

## Plain-Language Summary

The final method keeps the strongest empirical training signal from `rewind3`:
compact supervised targets extracted from the gold Vietnamese solution, filtered
for short, verifiable arithmetic chains, then trained with token-level weighting
around computation tokens and the final answer anchor.

Instead of assuming the last epoch is best, the notebook saves every epoch
checkpoint and evaluates them. It then performs a small decoding sweep on the
best checkpoint and uses the best validation decoding configuration for final
validation/test generation.

## Components

| Component | Implementation |
|---|---|
| Prompt | `PROMPT_TEMPLATE = "Câu hỏi: {q}\nGiải:\n"` |
| Target construction | `normalize_response()` / `_target` in `rewind3.ipynb` and `src/vn_gpt2_math/targets.py` |
| Filtering | Length, final-answer parseability, arithmetic verification, concise response filter |
| Loss | `WeightedLossTrainer` with computation and answer-anchor token weights |
| Evaluation | relative-error metric, score buckets 0/1/5/10 |
| Decoding | beam and self-consistency sweep; best observed config was SC with 15 samples at T=0.4 |
| Output cleanup | strips `huehue`, unit tails, and preserves supported LaTeX answers |

## Why This Was Chosen

Later target rewrites often made outputs cleaner but reduced correctness. The
final strategy therefore keeps the original rewind3 training distribution and
focuses on:

1. choosing the best epoch checkpoint;
2. using self-consistency decoding for arithmetic robustness;
3. preserving strong output cleanup without changing the scoring logic.

## Known Limitations

The remaining bottleneck is operation selection. Many wrong answers are still
fluent and extractable but mathematically incorrect. `SV` and `FOBAR` problem
types remain weak compared with rephrased GSM-style problems.
