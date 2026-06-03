# Experiment History

This file records the main directions we tried before settling on the final
notebook. Scores are local validation results, not hidden-test results.

## Summary Table

| Direction | What changed | Result |
|---|---|---|
| Starter SFT | full translated solutions | Repetition and weak answer extraction. |
| Answer-stub SFT | answer marker only | Cleaner formatting, poor arithmetic. |
| Equation-focused targets | compact equation snippets | Better numeric outputs, still wrong operations. |
| Tagged reasoning | `<think>/<answer>` style | Stable tags, no clear accuracy gain. |
| Local teacher preprocessing | Qwen/DeepSeek cleanup | Useful for analysis, not used in the final rule-safe run. |
| FDD-lite | exact sampled train candidates | Regressed after second-stage SFT. |
| GRPO/RLVR preflight | verifier reward on sampled answers | Reward too sparse for the runtime budget. |
| rewind family | compact targets + weighted loss | Best stable training setup. |
| self-consistency decoding | sample and vote final answers | Largest reliable inference-time gain. |

## Best Completed References

| Run | Validation examples | Score | Notes |
|---|---:|---:|---|
| rewind reference | 500 | 1682 / 5000 | 33.64%, `sc9_t04` fallback after sweep skip |
| final-testing reference | 1000 | 3285 / 10000 | 32.85%, best completed decode `sc21_t05` |

The 500-example run is useful for detailed failure analysis. The 1,000-example
run is the better reference for final reporting because it reduces validation
noise and includes a decoding comparison.

## Error Profile

The failure analysis consistently points to arithmetic mistakes as the main
source of lost score. Extraction failures are relatively small after answer
cleanup, so pushing harder on answer formatting is unlikely to move the score
much. The weak groups are `GSM_SV`, `GSM_FOBAR`, and `MATH_FOBAR`; the strong
groups are rephrased GSM problems.

## Data-Centric Work

The final data checks focused on problems that could silently corrupt training:

- broad Vietnamese number-word normalization;
- `phần N` being mistaken for a fraction when it means a section;
- LaTeX fraction flattening that changes operator precedence;
- Asymptote diagram code inside MATH records;
- duplicate or near-duplicate Vietnamese queries.

The reproducible audit entrypoint is:

```bash
python3 scripts/audit_dataset_quality.py \
  --data-dir /path/to/dataset-math \
  --output outputs/data_quality_report.json
```

The final notebook now also records a train-query dedup report and a set of
report figures under `/kaggle/working/report_artifacts/`.
