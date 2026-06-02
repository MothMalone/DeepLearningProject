# Experiment History

This log summarizes the main experimental directions. Scores are local
validation results and should be interpreted as development signals, not final
test performance.

| Family | Representative Config | Observation |
|---|---|---|
| Starter SFT | original long responses | Pipeline worked but generated repetitive, wrong reasoning. |
| Answer stub | `answer_stub`, answer marker | Improved extractability, but mostly learned answer format. |
| Equation focused | compact equation targets | Improved numeric pair rate; correctness remained low. |
| Tagged SFT | `<think>/<answer>` warmups | Stabilized tags but often guessed answers. |
| GRPO/RLVR | train-set verifier reward | Preflight showed sparse correctness reward; unsafe to scale. |
| Local LLM preprocessing | Qwen/DeepSeek teacher cleanup | Useful as a research direction, not used in final rule-safe Kaggle run. |
| FDD-lite/type routing | PoT-like targets | Some runs regressed due target mismatch and answer-only collapse. |
| rewind3 final | compact targets + weighted loss + SC decoding | Best local result; arithmetic errors remain dominant. |

## Best Observed rewind3 Run

```text
Validation subset : 500 examples
Raw score         : 1682 / 5000
Exact answers     : 157 / 500
Main gain         : self-consistency decoding over beam
```

## Failure Breakdown

The `VietnameseMathErrorAnalyzer` consistently identified arithmetic mistakes as
the dominant residual error. Formatting and extraction were mostly solved by the
final runs, so more answer-anchor pressure was not the right next move.
