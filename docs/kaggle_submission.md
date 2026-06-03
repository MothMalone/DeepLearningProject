# Kaggle Submission Notes

## Official Environment

- Internet: OFF
- Accelerator: GPU
- Inputs:
  - `kimanh2002/dataset-math`
  - `kimanh2002/nlphustgpt2-vietnamese`

## Required Outputs

The notebook must write:

```text
/kaggle/working/valid_output.json
/kaggle/working/valid_report.json
/kaggle/working/test_predictions.json
/kaggle/working/hpo_report.json
/kaggle/working/report_artifacts/
```

The test prediction schema is:

```json
[
  {
    "id": 0,
    "query_vi": "Question text",
    "type": "GSM_Rephrased",
    "model_output": "Giải: ... ####đáp án là: 42"
  }
]
```

Never include `response_vi`, `gold_answer`, reward fields, or validation-only
diagnostics in `test_predictions.json`.

## Safe Final Path

Run `notebooks/final_experiment.ipynb` inside Kaggle and submit the
Kaggle-generated output. Archived notebooks and Vast.ai scripts are development
history only.
