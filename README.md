# Vietnamese GPT-2 Math Fine-Tuning

This project fine-tunes a fixed Vietnamese GPT-2 model for Vietnamese math word
problems. The work started in Kaggle notebooks, then was organized into a small
training/evaluation package so the final run can be inspected and reproduced.

The final notebook is:

```text
notebooks/final_experiment.ipynb
```

Older notebooks are kept in `archive/notebooks/` for provenance, but they are not
part of the main project workflow.

## Task Setup

- Base model: `NlpHUST/gpt2-vietnamese`
- Data: `kimanh2002/dataset-math`
- Runtime target: Kaggle GPU notebook with Internet disabled
- Output file: `/kaggle/working/test_predictions.json`
- Metric: numeric answer score bucketed as 0/1/5/10 by relative error

The hidden-test path does not use labels, type routing, external solvers, or
test-time learning.

## Current Validation Reference

The strongest completed reference run used 1,000 validation examples and
self-consistency decoding:

| Setting | Value |
|---|---:|
| Validation score | 3285 / 10000 |
| Score percentage | 32.85% |
| Exact answers | 303 / 1000 |
| Best decoding | `sc21_t05` |

The earlier 500-example rewind run reached 1682 / 5000 (33.64%). Both result
sets are saved under `reports/final_results/`.

## Project Layout

```text
.
├── notebooks/
│   └── final_experiment.ipynb      # final Kaggle notebook
├── src/vn_gpt2_math/
│   ├── answers.py                  # answer extraction and numeric parsing
│   ├── targets.py                  # text cleanup and compact target creation
│   ├── dedup.py                    # train-query duplicate removal
│   ├── metrics.py                  # validation metric
│   ├── generation.py               # beam/self-consistency generation
│   ├── error_analysis.py           # failure taxonomy
│   └── reporting.py                # tables and report figures
├── scripts/
│   ├── audit_dataset_quality.py
│   └── build_report_artifacts.py
├── configs/
│   ├── final/
│   ├── decoding/
│   └── ablations/
├── reports/final_results/          # saved tables and charts
├── docs/                           # method notes and experiment history
├── tests/
├── train.py
├── evaluate.py
└── infer.py
```

## Reproduce the Final Notebook

In Kaggle:

1. Attach `kimanh2002/dataset-math`.
2. Attach `kimanh2002/nlphustgpt2-vietnamese`.
3. Turn Internet off and enable GPU.
4. Run `notebooks/final_experiment.ipynb`.

The notebook writes:

```text
/kaggle/working/valid_output.json
/kaggle/working/valid_report.json
/kaggle/working/test_predictions.json
/kaggle/working/hpo_report.json
/kaggle/working/report_artifacts/
```

## Local Checks

These commands do not train a model:

```bash
python3 -m unittest discover -s tests
python3 -m py_compile train.py evaluate.py infer.py scripts/build_report_artifacts.py
python3 scripts/build_report_artifacts.py --skip-patch
```

## What Worked

The most reliable gains came from the basic supervised setup rather than a
second-stage algorithm:

- compact arithmetic targets from the provided Vietnamese solutions;
- token-level weighting around computation tokens and final answer anchors;
- checkpoint selection by generated validation score;
- self-consistency decoding on the selected checkpoint.

GRPO and feedback-driven self-distillation were explored, but both were too
sparse or unstable for the available runtime. The remaining error profile is
mostly arithmetic: the model usually emits an extractable answer, but still
chooses the wrong operation or loses track of a variable chain.
