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

The safest official path is to run `rewind3.ipynb` inside Kaggle and submit the
Kaggle-generated notebook version. Vast.ai experiments are for development only.
