# Vast.ai Migration Workflow

Use Vast.ai for experiments, debugging, and ablations. Keep Kaggle as the official final run environment.

## 1. Kaggle Official Run

- Attach these Kaggle inputs:
  - `kimanh2002/dataset-math`
  - `kimanh2002/nlphustgpt2-vietnamese`
- Keep Internet OFF.
- Keep `ALLOW_KAGGLEHUB_DOWNLOAD = False`.
- Run the notebook with GPU.
- Required outputs are produced under `/kaggle/working`.

## 2. Vast.ai Experiment Run

Use one of two input strategies.

### Option A: kagglehub

Configure Kaggle credentials on the Vast instance, then set:

```python
ALLOW_KAGGLEHUB_DOWNLOAD = True
```

The notebook will call:

```python
kagglehub.dataset_download("kimanh2002/dataset-math")
kagglehub.dataset_download("kimanh2002/nlphustgpt2-vietnamese")
```

### Option B: explicit local folders

Download or copy the folders yourself, then launch the notebook with:

```bash
export DATA_DIR_OVERRIDE=/path/to/dataset-math
export MODEL_DIR_OVERRIDE=/path/to/nlphustgpt2-vietnamese
export WORKING_DIR=/workspace/kaggle_working
```

The training/evaluation code remains the same.

## 3. Recommended Experiment Order

1. `EXPERIMENT_MODE = "eda_only"` to verify paths, records, and answer extraction.
2. `EXPERIMENT_MODE = "small_debug"` with a few thousand samples.
3. `EXPERIMENT_MODE = "small_compare"` changing only one or two variables.
4. `EXPERIMENT_MODE = "full_train"` only after validation suggests the candidate is worth running.
5. `EXPERIMENT_MODE = "phase2_test"` only inside the official Kaggle setup when `test.json` is available.

## 4. Unattended Vast Experiment Runner

The repository includes:

```text
run_vast_experiments.sh
analyze_experiment_results.py
```

Default unattended run:

```bash
cd /workspace/vietnamese-gpt2-math
bash run_vast_experiments.sh
```

The default `RUN_SET=next_stage` runs targeted diagnostic ablations:

- `answer_stub` sanity check
- answer-line-only loss tests
- balanced/GSM-heavy sampling tests
- duplicate-control tests
- a conditional 50k/1000 candidate only if a non-answer-stub 20k/500 run beats the current 0.468 `score_10` reference

Preview without training:

```bash
DRY_RUN=true bash run_vast_experiments.sh
```

If paths are not auto-detected:

```bash
DATA_DIR_OVERRIDE=/workspace/dataset-math \
MODEL_DIR_OVERRIDE=/workspace/nlphustgpt2-vietnamese \
EXPERIMENTS_ROOT=/workspace/experiments \
RUN_SET=next_stage \
bash run_vast_experiments.sh
```

Outputs:

```text
/workspace/experiments/<experiment_name>/valid_output.json
/workspace/experiments/<experiment_name>/valid_report.json
/workspace/experiments/<experiment_name>/error_analysis.json
/workspace/experiments/<experiment_name>/run.log
/workspace/experiments/experiment_summary.json
/workspace/experiments/experiment_summary.md
```

Available run sets:

```text
quick     5k/200 smoke ablations
extended  5k/200 broader smoke ablations
hour      20k/500 focused ablations
next_stage  targeted A-D ablations, default
deep      next_stage plus extra easy/math-heavy ablations
grpo_debug  GRPO from an existing SFT checkpoint, 500 train examples
sft_then_grpo_debug  small SFT warmup followed by GRPO
grpo_hour  larger SFT-then-GRPO candidate
sft_then_grpo_hour  alias for the larger SFT-then-GRPO candidate
sft_tagged_preflight  SFT-only tagged answer warmup on GSM examples
sft_tagged_reasoning_from_answer_warmup  continue tagged SFT with equation/numeric reasoning
sft_tagged_minimal_reasoning_from_answer_warmup  continue tagged SFT with minimal equation traces
grpo_preflight_from_sft  rollout diagnostics from an existing tagged SFT checkpoint
grpo_preflight_from_sft_partial  same preflight with partial numeric GRPO reward enabled
preprocessed_sft_debug  train GPT-2 on a small locally distilled JSONL dataset
preprocessed_sft_20k  train GPT-2 on a larger locally distilled JSONL dataset
fdd_pot  Feedback-Driven Distillation-inspired Program-of-Thought candidate
```

Use `RUN_SET=deep` only when you are comfortable letting the instance run longer.
Set `RUN_BEST_CANDIDATE=true` to force a 50k/1000 scaled candidate even if no 20k/500 run beats the threshold.

GRPO dry run:

```bash
DRY_RUN=true RUN_SET=grpo_debug bash run_vast_experiments.sh
```

Tagged SFT warmup before GRPO:

```bash
RUN_SET=sft_tagged_preflight bash run_vast_experiments.sh
```

Second-stage tagged reasoning SFT from answer-tag warmup:

```bash
SFT_CKPT_DIR=/workspace/experiments/sft_tagged_answer_only_gsm_5k_200/gpt2_math_ckpt \
RUN_SET=sft_tagged_reasoning_from_answer_warmup \
bash run_vast_experiments.sh
```

Stricter minimal tagged reasoning SFT from answer-tag warmup:

```bash
SFT_CKPT_DIR=/workspace/experiments/sft_tagged_answer_only_gsm_20k_500/gpt2_math_ckpt \
RUN_SET=sft_tagged_minimal_reasoning_from_answer_warmup \
bash run_vast_experiments.sh
```

GRPO preflight from that checkpoint:

```bash
SFT_CKPT_DIR=/workspace/experiments/sft_tagged_equation_focused_gsm_20k_500_from_answer_warmup/gpt2_math_ckpt \
RUN_SET=grpo_preflight_from_sft \
bash run_vast_experiments.sh
```

GRPO preflight with partial numeric reward enabled:

```bash
SFT_CKPT_DIR=/workspace/experiments/sft_tagged_equation_minimal_gsm_20k_500_from_answer_warmup/gpt2_math_ckpt \
RUN_SET=grpo_preflight_from_sft_partial \
bash run_vast_experiments.sh
```

An explicit shell override wins over a run spec:

```bash
GRPO_USE_PARTIAL_NUMERIC_REWARD=false RUN_SET=grpo_preflight_from_sft_partial bash run_vast_experiments.sh
```

Small SFT then GRPO run:

```bash
RUN_SET=sft_then_grpo_debug bash run_vast_experiments.sh
```

GRPO from an existing SFT checkpoint:

```bash
SFT_CKPT_DIR=/workspace/experiments/some_sft_run/gpt2_math_ckpt \
RUN_SET=grpo_debug \
bash run_vast_experiments.sh
```

GRPO is experimental. It uses only train.json gold answers as a verifier and should be selected only if local validation improves.

FDD/PoT candidate run:

```bash
RUN_SET=fdd_pot \
DATA_DIR_OVERRIDE=/workspace/input/dataset-math \
MODEL_DIR_OVERRIDE=/workspace/input/nlphustgpt2-vietnamese \
EXPERIMENTS_ROOT=/workspace/experiments \
bash run_vast_experiments.sh
```

This run trains GPT-2 to emit short Python arithmetic programs plus `Đáp án là:`. The notebook can execute only the model-generated program to append the final numeric answer:

```bash
FDD_EXECUTE_POT_AT_INFERENCE=true RUN_SET=fdd_pot bash run_vast_experiments.sh
```

Disable execution if your final interpretation of the rules does not permit Program-of-Thought interpreter use:

```bash
FDD_EXECUTE_POT_AT_INFERENCE=false RUN_SET=fdd_pot bash run_vast_experiments.sh
```

## 5. Local LLM Preprocessing / Distillation

This is a Vast.ai/local experiment lane. It creates a cleaned training JSONL from `train.json` only, then the normal GPT-2 notebook can consume it with `TRAIN_DATA_OVERRIDE`. Do not preprocess `valid.json` or `test.json`.

Step 1: preprocess 500 GSM examples with a local open-source model:

```bash
python scripts/preprocess_with_local_llm.py \
  --input /workspace/input/dataset-math/train.json \
  --output /workspace/processed/qwen_math_distilled_gsm_500.jsonl \
  --report /workspace/processed/qwen_math_distilled_gsm_500_report.json \
  --model Qwen/Qwen2.5-Math-7B-Instruct \
  --max-examples 500 \
  --type-filter GSM \
  --load-in-4bit \
  --resume
```

Step 2: inspect the processed targets:

```bash
python scripts/inspect_preprocessed_dataset.py \
  --input /workspace/processed/qwen_math_distilled_gsm_500.jsonl \
  --n 10
```

Step 3: train GPT-2 on processed data:

```bash
TRAIN_DATA_OVERRIDE=/workspace/processed/qwen_math_distilled_gsm_500.jsonl \
RUN_SET=preprocessed_sft_debug \
DATA_DIR_OVERRIDE=/workspace/input/dataset-math \
MODEL_DIR_OVERRIDE=/workspace/input/nlphustgpt2-vietnamese \
EXPERIMENTS_ROOT=/workspace/experiments \
bash run_vast_experiments.sh
```

Scale only if validation and qualitative outputs justify it:

```bash
TRAIN_DATA_OVERRIDE=/workspace/processed/qwen_math_distilled_gsm_20k.jsonl \
RUN_SET=preprocessed_sft_20k \
DATA_DIR_OVERRIDE=/workspace/input/dataset-math \
MODEL_DIR_OVERRIDE=/workspace/input/nlphustgpt2-vietnamese \
EXPERIMENTS_ROOT=/workspace/experiments \
bash run_vast_experiments.sh
```

Current reference before this preprocessing lane is `score_10` around `0.58` on 500 validation examples. Continue scaling only if the 500-example debug run looks cleaner qualitatively and a 2k/20k processed run beats that local validation reference.

## 6. Submission Rule

Do not copy Vast-generated prediction files into Kaggle final submission. The final notebook must be run by Kaggle Save Version / Run All and generate its own `/kaggle/working/test_predictions.json`.
