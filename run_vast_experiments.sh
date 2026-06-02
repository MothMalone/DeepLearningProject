#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential Vast.ai experiment runner.
#
# Default goal: run a small, useful ablation set while you are away, archive each
# run in its own folder, and maintain aggregate JSON/Markdown analysis.
#
# Usage:
#   bash run_vast_experiments.sh
#
# Common overrides:
#   DATA_DIR_OVERRIDE=/workspace/dataset-math \
#   MODEL_DIR_OVERRIDE=/workspace/nlphustgpt2-vietnamese \
#   EXPERIMENTS_ROOT=/workspace/experiments \
#   RUN_SET=next_stage \
#   bash run_vast_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NOTEBOOK_PY="${NOTEBOOK_PY:-$SCRIPT_DIR/kaggle_vietnamese_gpt2_math_notebook.py}"
ANALYZE_PY="${ANALYZE_PY:-$SCRIPT_DIR/analyze_experiment_results.py}"

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/workspace/experiments}"
RUN_SET="${RUN_SET:-next_stage}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-false}"
DRY_RUN="${DRY_RUN:-false}"
RUN_BEST_CANDIDATE="${RUN_BEST_CANDIDATE:-auto}"
BEST_CANDIDATE_THRESHOLD="${BEST_CANDIDATE_THRESHOLD:-0.468}"

DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-}"
MODEL_DIR_OVERRIDE="${MODEL_DIR_OVERRIDE:-}"
TRAIN_DATA_OVERRIDE="${TRAIN_DATA_OVERRIDE:-}"
PREPROCESSED_TARGET_FIELD="${PREPROCESSED_TARGET_FIELD:-target_direct}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-}"
GRPO_USE_PARTIAL_NUMERIC_REWARD_OVERRIDE="${GRPO_USE_PARTIAL_NUMERIC_REWARD:-}"

# Conservative defaults for an unattended Vast run. Increase these after the
# runner proves stable on your instance.
BASE_MAX_TRAIN_SAMPLES="${BASE_MAX_TRAIN_SAMPLES:-5000}"
BASE_MAX_VALID_SAMPLES="${BASE_MAX_VALID_SAMPLES:-200}"
HOUR_MAX_TRAIN_SAMPLES="${HOUR_MAX_TRAIN_SAMPLES:-20000}"
HOUR_MAX_VALID_SAMPLES="${HOUR_MAX_VALID_SAMPLES:-500}"
DEEP_MAX_TRAIN_SAMPLES="${DEEP_MAX_TRAIN_SAMPLES:-50000}"
DEEP_MAX_VALID_SAMPLES="${DEEP_MAX_VALID_SAMPLES:-1000}"
BASE_EPOCHS="${BASE_EPOCHS:-1}"
BASE_MAX_LENGTH="${BASE_MAX_LENGTH:-384}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-96}"
BASE_PER_DEVICE_BATCH_SIZE="${BASE_PER_DEVICE_BATCH_SIZE:-4}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-8}"
BASE_GEN_BATCH_SIZE="${BASE_GEN_BATCH_SIZE:-8}"
BASE_SEED="${BASE_SEED:-42}"
BASE_TRAIN_TYPE_FILTER="${BASE_TRAIN_TYPE_FILTER:-}"
BASE_TAGGED_REASONING_MAX_CHARS="${BASE_TAGGED_REASONING_MAX_CHARS:-700}"
BASE_TAGGED_REASONING_MAX_PARTS="${BASE_TAGGED_REASONING_MAX_PARTS:-6}"
BASE_FDD_EXECUTE_POT_AT_INFERENCE="${BASE_FDD_EXECUTE_POT_AT_INFERENCE:-true}"
BASE_FDD_MAX_EQUATION_STEPS="${BASE_FDD_MAX_EQUATION_STEPS:-5}"
BASE_FDD_PROGRAM_MAX_LINES="${BASE_FDD_PROGRAM_MAX_LINES:-24}"
BASE_FDD_KEEP_CODE_IN_OUTPUT="${BASE_FDD_KEEP_CODE_IN_OUTPUT:-true}"

BASE_TRAIN_STAGE="${BASE_TRAIN_STAGE:-sft}"
BASE_GRPO_MAX_TRAIN_SAMPLES="${BASE_GRPO_MAX_TRAIN_SAMPLES:-500}"
BASE_GRPO_ITERATIONS="${BASE_GRPO_ITERATIONS:-1}"
BASE_GRPO_ROLLOUT_BATCH_SIZE="${BASE_GRPO_ROLLOUT_BATCH_SIZE:-4}"
BASE_GRPO_NUM_GENERATIONS="${BASE_GRPO_NUM_GENERATIONS:-4}"
BASE_GRPO_BUFFER_SIZE="${BASE_GRPO_BUFFER_SIZE:-128}"
BASE_GRPO_MINIBATCH_SIZE="${BASE_GRPO_MINIBATCH_SIZE:-4}"
BASE_GRPO_EPOCHS_PER_BUFFER="${BASE_GRPO_EPOCHS_PER_BUFFER:-1}"
BASE_GRPO_LR="${BASE_GRPO_LR:-1e-6}"
BASE_GRPO_CLIP_EPS="${BASE_GRPO_CLIP_EPS:-0.2}"
BASE_GRPO_TEMPERATURE="${BASE_GRPO_TEMPERATURE:-0.9}"
BASE_GRPO_TOP_P="${BASE_GRPO_TOP_P:-0.95}"
BASE_GRPO_TOP_K="${BASE_GRPO_TOP_K:-0}"
BASE_GRPO_GRAD_ACCUM="${BASE_GRPO_GRAD_ACCUM:-1}"
BASE_GRPO_MAX_GRAD_NORM="${BASE_GRPO_MAX_GRAD_NORM:-1.0}"
BASE_GRPO_CORRECTNESS_WEIGHT="${BASE_GRPO_CORRECTNESS_WEIGHT:-0.95}"
BASE_GRPO_FORMAT_WEIGHT="${BASE_GRPO_FORMAT_WEIGHT:-0.05}"
BASE_GRPO_USE_PARTIAL_NUMERIC_REWARD="${BASE_GRPO_USE_PARTIAL_NUMERIC_REWARD:-false}"
BASE_GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER="${BASE_GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER:-true}"
BASE_GRPO_PREFLIGHT_ONLY="${BASE_GRPO_PREFLIGHT_ONLY:-false}"

if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$EXPERIMENTS_ROOT"
fi

if [[ ! -f "$NOTEBOOK_PY" ]]; then
  echo "Notebook Python entrypoint not found: $NOTEBOOK_PY" >&2
  exit 1
fi

if [[ ! -f "$ANALYZE_PY" ]]; then
  echo "Analysis script not found: $ANALYZE_PY" >&2
  exit 1
fi

if [[ -z "$DATA_DIR_OVERRIDE" ]]; then
  for p in /workspace/input/dataset-math /workspace/dataset-math /workspace/working/dataset-math; do
    if [[ -d "$p" ]]; then
      DATA_DIR_OVERRIDE="$p"
      break
    fi
  done
fi

if [[ -z "$MODEL_DIR_OVERRIDE" ]]; then
  for p in /workspace/input/nlphustgpt2-vietnamese /workspace/nlphustgpt2-vietnamese /workspace/working/nlphustgpt2-vietnamese; do
    if [[ -d "$p" ]]; then
      MODEL_DIR_OVERRIDE="$p"
      break
    fi
  done
fi

echo "NOTEBOOK_PY=$NOTEBOOK_PY"
echo "ANALYZE_PY=$ANALYZE_PY"
echo "EXPERIMENTS_ROOT=$EXPERIMENTS_ROOT"
echo "RUN_SET=$RUN_SET"
echo "OVERWRITE_RUNS=$OVERWRITE_RUNS"
echo "DRY_RUN=$DRY_RUN"
echo "RUN_BEST_CANDIDATE=$RUN_BEST_CANDIDATE"
echo "BEST_CANDIDATE_THRESHOLD=$BEST_CANDIDATE_THRESHOLD"
echo "DATA_DIR_OVERRIDE=${DATA_DIR_OVERRIDE:-<auto/kagglehub>}"
echo "MODEL_DIR_OVERRIDE=${MODEL_DIR_OVERRIDE:-<auto/kagglehub>}"
echo "TRAIN_DATA_OVERRIDE=${TRAIN_DATA_OVERRIDE:-<none>}"
echo "PREPROCESSED_TARGET_FIELD=$PREPROCESSED_TARGET_FIELD"
echo "SFT_CKPT_DIR=${SFT_CKPT_DIR:-<none>}"
echo "GRPO_USE_PARTIAL_NUMERIC_REWARD_OVERRIDE=${GRPO_USE_PARTIAL_NUMERIC_REWARD_OVERRIDE:-<spec/default>}"

# Format:
# name|train_style|prompt_style|loss_style|sampling_style|dedup|decoding_style|lr|max_length|max_new_tokens|filter_non_numeric_gold|max_train|max_valid|epochs|optional_train_type_filter|optional_tagged_reasoning_max_chars|optional_tagged_reasoning_max_parts
QUICK_EXPERIMENTS=(
  "answer_stub_answer_marker_5k_200|answer_stub|answer_marker|full_target|natural|false|short_greedy|5e-5|256|32|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
  "equation_answer_marker_answerloss_5k_200|equation_focused|answer_marker|answer_line_only|natural|false|short_greedy|5e-5|384|64|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
  "standardized_answer_marker_answerloss_5k_200|standardized|answer_marker|answer_line_only|natural|false|short_greedy|5e-5|384|96|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
)

EXTENDED_EXPERIMENTS=(
  "${QUICK_EXPERIMENTS[@]}"
  "equation_answer_marker_fulltarget_5k_200|equation_focused|answer_marker|full_target|natural|false|short_greedy|5e-5|384|64|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
  "short_answer_marker_5k_200|short_solution|answer_marker|full_target|natural|false|short_greedy|5e-5|384|96|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
  "equation_answer_marker_beam2_5k_200|equation_focused|answer_marker|answer_line_only|natural|false|beam2|5e-5|384|64|false|$BASE_MAX_TRAIN_SAMPLES|$BASE_MAX_VALID_SAMPLES|$BASE_EPOCHS"
)

HOUR_EXPERIMENTS=(
  "equation_answer_marker_balanced_20k_500|equation_focused|answer_marker|answer_line_only|balanced_type|false|short_greedy|5e-5|384|64|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
  "equation_answer_marker_gsmheavy_20k_500|equation_focused|answer_marker|answer_line_only|gsm_heavy|false|short_greedy|5e-5|384|64|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
  "answer_stub_answer_marker_balanced_20k_500|answer_stub|answer_marker|full_target|balanced_type|false|short_greedy|5e-5|256|32|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
  "equation_answer_marker_dedup_20k_500|equation_focused|answer_marker|answer_line_only|natural|true|short_greedy|5e-5|384|64|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
  "answer_stub_answer_marker_dedup_20k_500|answer_stub|answer_marker|full_target|natural|true|short_greedy|5e-5|256|32|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
)

NEXT_STAGE_EXPERIMENTS=(
  "${QUICK_EXPERIMENTS[@]}"
  "${HOUR_EXPERIMENTS[@]}"
)

DEEP_EXPERIMENTS=(
  "${NEXT_STAGE_EXPERIMENTS[@]}"
  "equation_answer_marker_mathheavy_20k_500|equation_focused|answer_marker|answer_line_only|math_heavy|false|short_greedy|5e-5|384|64|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
  "equation_answer_marker_easyfirst_20k_500|equation_focused|answer_marker|answer_line_only|easy_first|false|short_greedy|5e-5|384|64|false|$HOUR_MAX_TRAIN_SAMPLES|$HOUR_MAX_VALID_SAMPLES|1"
)

# GRPO format:
# name|train_stage|train_style|prompt_style|loss_style|sampling_style|dedup|decoding_style|lr|max_length|max_new_tokens|filter_non_numeric_gold|max_train|max_valid|epochs|grpo_max_train|grpo_iterations|grpo_rollout_batch|grpo_num_generations|grpo_buffer_size|grpo_minibatch|grpo_lr|grpo_temperature|grpo_top_p|grpo_top_k|grpo_partial_reward|train_type_filter|grpo_preflight_only|optional_tagged_reasoning_max_chars|optional_tagged_reasoning_max_parts
GRPO_DEBUG_EXPERIMENTS=(
  "grpo_from_existing_sft_equation_500|grpo|equation_focused|answer_marker|answer_line_only|natural|false|short_greedy|5e-5|384|64|false|5000|200|1|500|1|4|4|128|4|1e-6|0.9|0.95|0|false"
)

SFT_THEN_GRPO_DEBUG_EXPERIMENTS=(
  "sft_then_grpo_tagged_short_500|sft_then_grpo|tagged_short_solution|answer_marker|full_target|natural|false|short_greedy|5e-5|384|64|false|5000|200|1|500|1|4|4|128|4|1e-6|0.9|0.95|0|false"
  "sft_then_grpo_equation_500|sft_then_grpo|equation_focused|answer_marker|answer_line_only|natural|false|short_greedy|5e-5|384|64|false|5000|200|1|500|1|4|4|128|4|1e-6|0.9|0.95|0|false"
)

GRPO_HOUR_EXPERIMENTS=(
  "sft_then_grpo_tagged_short_2k|sft_then_grpo|tagged_short_solution|answer_marker|full_target|balanced_type|false|short_greedy|5e-5|384|64|false|20000|500|1|2000|2|4|4|512|4|1e-6|0.9|0.95|0|false"
)

SFT_THEN_GRPO_HOUR_EXPERIMENTS=(
  "${GRPO_HOUR_EXPERIMENTS[@]}"
)

SFT_TAGGED_PREFLIGHT_EXPERIMENTS=(
  "sft_tagged_answer_only_gsm_5k_200|tagged_answer_only|answer_marker|full_target|natural|false|short_greedy|5e-5|384|64|false|5000|200|1|GSM"
)

SFT_TAGGED_REASONING_FROM_ANSWER_WARMUP_EXPERIMENTS=(
  "sft_tagged_equation_focused_gsm_20k_500_from_answer_warmup|tagged_equation_focused|answer_marker|full_target|natural|false|short_greedy|5e-5|512|128|false|20000|500|1|GSM"
)

SFT_TAGGED_MINIMAL_REASONING_FROM_ANSWER_WARMUP_EXPERIMENTS=(
  "sft_tagged_equation_minimal_gsm_20k_500_from_answer_warmup|tagged_equation_minimal|answer_marker|full_target|natural|false|short_greedy|5e-5|384|80|false|20000|500|1|GSM|300|3"
)

GRPO_PREFLIGHT_FROM_SFT_EXPERIMENTS=(
  "grpo_preflight_from_sft_tagged_reasoning_gsm_500|grpo|tagged_equation_focused|answer_marker|full_target|natural|false|short_greedy|5e-5|512|128|false|5000|200|1|500|1|4|4|128|4|1e-6|0.9|0.95|0|false|GSM|true"
)

GRPO_PREFLIGHT_FROM_SFT_PARTIAL_EXPERIMENTS=(
  "grpo_preflight_from_sft_partial_tagged_minimal_gsm_500|grpo|tagged_equation_minimal|answer_marker|full_target|natural|false|short_greedy|5e-5|384|80|false|5000|200|1|500|1|4|4|128|4|1e-6|0.9|0.95|0|true|GSM|true|300|3"
)

PREPROCESSED_SFT_DEBUG_EXPERIMENTS=(
  "qwen_distilled_gsm_500_direct|preprocessed_target|direct_answer_marker|full_target|natural|false|short_greedy|5e-5|384|64|false|500|500|1"
)

PREPROCESSED_SFT_20K_EXPERIMENTS=(
  "qwen_distilled_gsm_20k_direct|preprocessed_target|direct_answer_marker|full_target|natural|false|short_greedy|5e-5|384|64|false|20000|500|1"
)

FDD_POT_EXPERIMENTS=(
  "fdd_pot_gsm_20k_500|fdd_pot|pot_instruction|full_target|gsm_heavy|false|short_greedy|5e-5|512|160|true|20000|500|2|GSM"
  "fdd_pot_all_50k_1000|fdd_pot|pot_instruction|full_target|natural|false|short_greedy|5e-5|512|160|true|50000|1000|2"
)

if [[ "$RUN_SET" == "quick" ]]; then
  EXPERIMENTS=("${QUICK_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "extended" ]]; then
  EXPERIMENTS=("${EXTENDED_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "hour" ]]; then
  EXPERIMENTS=("${HOUR_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "next_stage" ]]; then
  EXPERIMENTS=("${NEXT_STAGE_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "deep" ]]; then
  EXPERIMENTS=("${DEEP_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "grpo_debug" ]]; then
  EXPERIMENTS=("${GRPO_DEBUG_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "sft_then_grpo_debug" ]]; then
  EXPERIMENTS=("${SFT_THEN_GRPO_DEBUG_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "grpo_hour" ]]; then
  EXPERIMENTS=("${GRPO_HOUR_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "sft_then_grpo_hour" ]]; then
  EXPERIMENTS=("${SFT_THEN_GRPO_HOUR_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "sft_tagged_preflight" ]]; then
  EXPERIMENTS=("${SFT_TAGGED_PREFLIGHT_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "sft_tagged_reasoning_from_answer_warmup" ]]; then
  EXPERIMENTS=("${SFT_TAGGED_REASONING_FROM_ANSWER_WARMUP_EXPERIMENTS[@]}")
  if [[ -z "$SFT_CKPT_DIR" ]]; then
    SFT_CKPT_DIR="/workspace/experiments/sft_tagged_answer_only_gsm_5k_200/gpt2_math_ckpt"
    echo "SFT_CKPT_DIR not set; defaulting to $SFT_CKPT_DIR for $RUN_SET"
  fi
elif [[ "$RUN_SET" == "sft_tagged_minimal_reasoning_from_answer_warmup" ]]; then
  EXPERIMENTS=("${SFT_TAGGED_MINIMAL_REASONING_FROM_ANSWER_WARMUP_EXPERIMENTS[@]}")
  if [[ -z "$SFT_CKPT_DIR" ]]; then
    SFT_CKPT_DIR="/workspace/experiments/sft_tagged_answer_only_gsm_20k_500/gpt2_math_ckpt"
    echo "SFT_CKPT_DIR not set; defaulting to $SFT_CKPT_DIR for $RUN_SET"
  fi
elif [[ "$RUN_SET" == "grpo_preflight_from_sft" ]]; then
  EXPERIMENTS=("${GRPO_PREFLIGHT_FROM_SFT_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "grpo_preflight_from_sft_partial" ]]; then
  EXPERIMENTS=("${GRPO_PREFLIGHT_FROM_SFT_PARTIAL_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "preprocessed_sft_debug" ]]; then
  EXPERIMENTS=("${PREPROCESSED_SFT_DEBUG_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "preprocessed_sft_20k" ]]; then
  EXPERIMENTS=("${PREPROCESSED_SFT_20K_EXPERIMENTS[@]}")
elif [[ "$RUN_SET" == "fdd_pot" ]]; then
  EXPERIMENTS=("${FDD_POT_EXPERIMENTS[@]}")
else
  echo "Unknown RUN_SET=$RUN_SET. Use quick, extended, hour, next_stage, deep, grpo_debug, sft_then_grpo_debug, grpo_hour, sft_then_grpo_hour, sft_tagged_preflight, sft_tagged_reasoning_from_answer_warmup, sft_tagged_minimal_reasoning_from_answer_warmup, grpo_preflight_from_sft, grpo_preflight_from_sft_partial, preprocessed_sft_debug, preprocessed_sft_20k, or fdd_pot." >&2
  exit 1
fi

echo "Planned experiments (${#EXPERIMENTS[@]}):"
for spec in "${EXPERIMENTS[@]}"; do
  echo "  - $spec"
done

run_one() {
  local spec="$1"
  local fields
  IFS='|' read -r -a fields <<< "$spec"

  local name train_stage train_style prompt_style loss_style sampling_style dedup decoding_style lr max_length max_new_tokens filter_non_numeric max_train max_valid epochs train_type_filter
  local tagged_reasoning_max_chars tagged_reasoning_max_parts
  local grpo_max_train grpo_iterations grpo_rollout_batch grpo_num_generations grpo_buffer_size grpo_minibatch grpo_lr grpo_temperature grpo_top_p grpo_top_k grpo_partial_reward grpo_preflight_only effective_grpo_partial_reward

  if [[ "${#fields[@]}" -ge 14 && "${#fields[@]}" -le 17 ]]; then
    name="${fields[0]}"
    train_stage="$BASE_TRAIN_STAGE"
    train_style="${fields[1]}"
    prompt_style="${fields[2]}"
    loss_style="${fields[3]}"
    sampling_style="${fields[4]}"
    dedup="${fields[5]}"
    decoding_style="${fields[6]}"
    lr="${fields[7]}"
    max_length="${fields[8]}"
    max_new_tokens="${fields[9]}"
    filter_non_numeric="${fields[10]}"
    max_train="${fields[11]}"
    max_valid="${fields[12]}"
    epochs="${fields[13]}"
    train_type_filter="${fields[14]:-$BASE_TRAIN_TYPE_FILTER}"
    tagged_reasoning_max_chars="${fields[15]:-$BASE_TAGGED_REASONING_MAX_CHARS}"
    tagged_reasoning_max_parts="${fields[16]:-$BASE_TAGGED_REASONING_MAX_PARTS}"
    grpo_max_train="$BASE_GRPO_MAX_TRAIN_SAMPLES"
    grpo_iterations="$BASE_GRPO_ITERATIONS"
    grpo_rollout_batch="$BASE_GRPO_ROLLOUT_BATCH_SIZE"
    grpo_num_generations="$BASE_GRPO_NUM_GENERATIONS"
    grpo_buffer_size="$BASE_GRPO_BUFFER_SIZE"
    grpo_minibatch="$BASE_GRPO_MINIBATCH_SIZE"
    grpo_lr="$BASE_GRPO_LR"
    grpo_temperature="$BASE_GRPO_TEMPERATURE"
    grpo_top_p="$BASE_GRPO_TOP_P"
    grpo_top_k="$BASE_GRPO_TOP_K"
    grpo_partial_reward="$BASE_GRPO_USE_PARTIAL_NUMERIC_REWARD"
    grpo_preflight_only="$BASE_GRPO_PREFLIGHT_ONLY"
  elif [[ "${#fields[@]}" -ge 26 ]]; then
    name="${fields[0]}"
    train_stage="${fields[1]}"
    train_style="${fields[2]}"
    prompt_style="${fields[3]}"
    loss_style="${fields[4]}"
    sampling_style="${fields[5]}"
    dedup="${fields[6]}"
    decoding_style="${fields[7]}"
    lr="${fields[8]}"
    max_length="${fields[9]}"
    max_new_tokens="${fields[10]}"
    filter_non_numeric="${fields[11]}"
    max_train="${fields[12]}"
    max_valid="${fields[13]}"
    epochs="${fields[14]}"
    grpo_max_train="${fields[15]}"
    grpo_iterations="${fields[16]}"
    grpo_rollout_batch="${fields[17]}"
    grpo_num_generations="${fields[18]}"
    grpo_buffer_size="${fields[19]}"
    grpo_minibatch="${fields[20]}"
    grpo_lr="${fields[21]}"
    grpo_temperature="${fields[22]}"
    grpo_top_p="${fields[23]}"
    grpo_top_k="${fields[24]}"
    grpo_partial_reward="${fields[25]}"
    train_type_filter="${fields[26]:-$BASE_TRAIN_TYPE_FILTER}"
    grpo_preflight_only="${fields[27]:-$BASE_GRPO_PREFLIGHT_ONLY}"
    tagged_reasoning_max_chars="${fields[28]:-$BASE_TAGGED_REASONING_MAX_CHARS}"
    tagged_reasoning_max_parts="${fields[29]:-$BASE_TAGGED_REASONING_MAX_PARTS}"
  else
    echo "Invalid experiment spec with ${#fields[@]} fields: $spec" >&2
    return 2
  fi

  if [[ "$train_style" == "fdd_pot" && "$train_stage" == "sft" ]]; then
    train_stage="fdd_pot"
  fi

  effective_grpo_partial_reward="$grpo_partial_reward"
  if [[ -n "$GRPO_USE_PARTIAL_NUMERIC_REWARD_OVERRIDE" ]]; then
    effective_grpo_partial_reward="$GRPO_USE_PARTIAL_NUMERIC_REWARD_OVERRIDE"
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo
    echo "DRY RUN parsed experiment: $name"
    echo "  train_stage=$train_stage train_style=$train_style prompt_style=$prompt_style loss_style=$loss_style"
    echo "  sampling_style=$sampling_style train_type_filter=$train_type_filter dedup=$dedup decoding_style=$decoding_style"
    echo "  tagged_reasoning_max_chars=$tagged_reasoning_max_chars tagged_reasoning_max_parts=$tagged_reasoning_max_parts"
    echo "  max_train=$max_train max_valid=$max_valid max_length=$max_length max_new_tokens=$max_new_tokens lr=$lr"
    echo "  grpo_max_train=$grpo_max_train grpo_iterations=$grpo_iterations grpo_rollout_batch=$grpo_rollout_batch"
    echo "  grpo_num_generations=$grpo_num_generations grpo_buffer_size=$grpo_buffer_size grpo_minibatch=$grpo_minibatch"
    echo "  grpo_lr=$grpo_lr grpo_temperature=$grpo_temperature grpo_top_p=$grpo_top_p grpo_top_k=$grpo_top_k"
    echo "  grpo_partial_reward=$effective_grpo_partial_reward grpo_preflight_only=$grpo_preflight_only sft_ckpt_dir=${SFT_CKPT_DIR:-}"
    echo "  train_data_override=${TRAIN_DATA_OVERRIDE:-} preprocessed_target_field=$PREPROCESSED_TARGET_FIELD"
    return 0
  fi

  local run_dir="$EXPERIMENTS_ROOT/$name"
  if [[ -e "$run_dir" && "$OVERWRITE_RUNS" != "true" ]]; then
    local i=2
    while [[ -e "${run_dir}_r${i}" ]]; do
      i=$((i + 1))
    done
    run_dir="${run_dir}_r${i}"
  fi
  local log_path="$run_dir/run.log"
  mkdir -p "$run_dir"

  echo
  echo "===================================================================================================="
  echo "Starting experiment: $name"
  echo "===================================================================================================="
  echo "Run dir: $run_dir"

  cp "$NOTEBOOK_PY" "$run_dir/notebook_entrypoint_snapshot.py"
  cp "$ANALYZE_PY" "$run_dir/analyze_experiment_results_snapshot.py"

  cat > "$run_dir/run_config.json" <<JSON
{
  "name": "$name",
  "train_stage": "$train_stage",
  "train_style": "$train_style",
  "prompt_style": "$prompt_style",
  "loss_style": "$loss_style",
  "sampling_style": "$sampling_style",
  "train_type_filter": "$train_type_filter",
  "dedup_train_questions": "$dedup",
  "tagged_reasoning_max_chars": "$tagged_reasoning_max_chars",
  "tagged_reasoning_max_parts": "$tagged_reasoning_max_parts",
  "fdd_execute_pot_at_inference": "$BASE_FDD_EXECUTE_POT_AT_INFERENCE",
  "fdd_max_equation_steps": "$BASE_FDD_MAX_EQUATION_STEPS",
  "fdd_program_max_lines": "$BASE_FDD_PROGRAM_MAX_LINES",
  "fdd_keep_code_in_output": "$BASE_FDD_KEEP_CODE_IN_OUTPUT",
  "decoding_style": "$decoding_style",
  "lr": "$lr",
  "max_length": "$max_length",
  "max_new_tokens": "$max_new_tokens",
  "filter_non_numeric_gold": "$filter_non_numeric",
  "max_train_samples": "$max_train",
  "max_valid_samples": "$max_valid",
  "epochs": "$epochs",
  "grpo_max_train_samples": "$grpo_max_train",
  "grpo_iterations": "$grpo_iterations",
  "grpo_rollout_batch_size": "$grpo_rollout_batch",
  "grpo_num_generations": "$grpo_num_generations",
  "grpo_buffer_size": "$grpo_buffer_size",
  "grpo_minibatch_size": "$grpo_minibatch",
  "grpo_lr": "$grpo_lr",
  "grpo_temperature": "$grpo_temperature",
  "grpo_top_p": "$grpo_top_p",
  "grpo_top_k": "$grpo_top_k",
  "grpo_use_partial_numeric_reward": "$effective_grpo_partial_reward",
  "grpo_gate_format_reward_on_numeric_answer": "$BASE_GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER",
  "grpo_preflight_only": "$grpo_preflight_only",
  "sft_ckpt_dir": "${SFT_CKPT_DIR:-}",
  "data_dir_override": "$DATA_DIR_OVERRIDE",
  "model_dir_override": "$MODEL_DIR_OVERRIDE",
  "train_data_override": "$TRAIN_DATA_OVERRIDE",
  "preprocessed_target_field": "$PREPROCESSED_TARGET_FIELD",
  "working_dir": "$run_dir"
}
JSON

  set +e
  (
    export RUN_ENV=vast
    export EXPERIMENT_MODE=small_debug
    export TRAIN_STAGE="$train_stage"
    export TRAIN_STYLE="$train_style"
    export PROMPT_STYLE="$prompt_style"
    export LOSS_STYLE="$loss_style"
    export SAMPLING_STYLE="$sampling_style"
    export TRAIN_TYPE_FILTER="$train_type_filter"
    export DEDUP_TRAIN_QUESTIONS="$dedup"
    export TAGGED_REASONING_MAX_CHARS="$tagged_reasoning_max_chars"
    export TAGGED_REASONING_MAX_PARTS="$tagged_reasoning_max_parts"
    export FDD_EXECUTE_POT_AT_INFERENCE="$BASE_FDD_EXECUTE_POT_AT_INFERENCE"
    export FDD_MAX_EQUATION_STEPS="$BASE_FDD_MAX_EQUATION_STEPS"
    export FDD_PROGRAM_MAX_LINES="$BASE_FDD_PROGRAM_MAX_LINES"
    export FDD_KEEP_CODE_IN_OUTPUT="$BASE_FDD_KEEP_CODE_IN_OUTPUT"
    export DECODING_STYLE="$decoding_style"
    export MAX_TRAIN_SAMPLES="$max_train"
    export MAX_VALID_SAMPLES="$max_valid"
    export EPOCHS="$epochs"
    export MAX_LENGTH="$max_length"
    export PER_DEVICE_BATCH_SIZE="$BASE_PER_DEVICE_BATCH_SIZE"
    export GRAD_ACCUM="$BASE_GRAD_ACCUM"
    export LR="$lr"
    export MAX_NEW_TOKENS="$max_new_tokens"
    export GEN_BATCH_SIZE="$BASE_GEN_BATCH_SIZE"
    export SEED="$BASE_SEED"
    export FILTER_NON_NUMERIC_GOLD="$filter_non_numeric"
    export FILTER_NO_ANSWER=false
    export RUN_BASELINE_FIRST=false
    export RUN_TRAIN=true
    export RUN_VALIDATION=true
    export RUN_TEST=false
    if [[ "$train_stage" == "grpo" || "$train_stage" == "sft_then_grpo" ]]; then
      export RUN_GRPO=true
    else
      export RUN_GRPO=false
    fi
    export SFT_CKPT_DIR="${SFT_CKPT_DIR:-}"
    export GRPO_MAX_TRAIN_SAMPLES="$grpo_max_train"
    export GRPO_ITERATIONS="$grpo_iterations"
    export GRPO_ROLLOUT_BATCH_SIZE="$grpo_rollout_batch"
    export GRPO_NUM_GENERATIONS="$grpo_num_generations"
    export GRPO_BUFFER_SIZE="$grpo_buffer_size"
    export GRPO_MINIBATCH_SIZE="$grpo_minibatch"
    export GRPO_EPOCHS_PER_BUFFER="$BASE_GRPO_EPOCHS_PER_BUFFER"
    export GRPO_LR="$grpo_lr"
    export GRPO_CLIP_EPS="$BASE_GRPO_CLIP_EPS"
    export GRPO_MAX_NEW_TOKENS="$max_new_tokens"
    export GRPO_TEMPERATURE="$grpo_temperature"
    export GRPO_TOP_P="$grpo_top_p"
    export GRPO_TOP_K="$grpo_top_k"
    export GRPO_GRAD_ACCUM="$BASE_GRPO_GRAD_ACCUM"
    export GRPO_MAX_GRAD_NORM="$BASE_GRPO_MAX_GRAD_NORM"
    export GRPO_CORRECTNESS_WEIGHT="$BASE_GRPO_CORRECTNESS_WEIGHT"
    export GRPO_FORMAT_WEIGHT="$BASE_GRPO_FORMAT_WEIGHT"
    export GRPO_USE_PARTIAL_NUMERIC_REWARD="$effective_grpo_partial_reward"
    export GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER="$BASE_GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER"
    export GRPO_PREFLIGHT_ONLY="$grpo_preflight_only"
    export POSTPROCESS_APPEND_LAST_NUMBER=true
    export WORKING_DIR="$run_dir"
    export ALLOW_KAGGLEHUB_DOWNLOAD="${ALLOW_KAGGLEHUB_DOWNLOAD:-false}"
    export TRAIN_DATA_OVERRIDE="$TRAIN_DATA_OVERRIDE"
    export PREPROCESSED_TARGET_FIELD="$PREPROCESSED_TARGET_FIELD"
    if [[ -n "$DATA_DIR_OVERRIDE" ]]; then export DATA_DIR_OVERRIDE; fi
    if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then export MODEL_DIR_OVERRIDE; fi

    python -u "$NOTEBOOK_PY"
  ) 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e

  python - "$run_dir" "$status" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
status = int(sys.argv[2])
payload = {
    "exit_code": status,
    "ok": status == 0,
    "finished_at_unix": time.time(),
}
(run_dir / "run_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  python "$ANALYZE_PY" --root "$EXPERIMENTS_ROOT" \
    --out-json "$EXPERIMENTS_ROOT/experiment_summary.json" \
    --out-md "$EXPERIMENTS_ROOT/experiment_summary.md" || true

  if [[ "$status" -ne 0 ]]; then
    echo "Experiment failed: $name (exit $status). Continuing to next run." >&2
  else
    echo "Finished experiment: $name"
  fi
}

for spec in "${EXPERIMENTS[@]}"; do
  run_one "$spec"
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "DRY_RUN=true; parsed all planned experiments without training."
  exit 0
fi

if [[ "$RUN_SET" == "next_stage" && "$RUN_BEST_CANDIDATE" != "false" ]]; then
  python "$ANALYZE_PY" --root "$EXPERIMENTS_ROOT" \
    --out-json "$EXPERIMENTS_ROOT/experiment_summary.json" \
    --out-md "$EXPERIMENTS_ROOT/experiment_summary.md" || true

  best_spec="$(
    python - "$EXPERIMENTS_ROOT/experiment_summary.json" "$BEST_CANDIDATE_THRESHOLD" "$RUN_BEST_CANDIDATE" "$DEEP_MAX_TRAIN_SAMPLES" "$DEEP_MAX_VALID_SAMPLES" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
threshold = float(sys.argv[2])
mode = sys.argv[3]
deep_train = sys.argv[4]
deep_valid = sys.argv[5]

if not summary_path.exists():
    sys.exit(0)

payload = json.loads(summary_path.read_text(encoding="utf-8"))
runs = payload.get("runs", [])

def score_key(run):
    s = run.get("summary", {})
    n = int(s.get("n") or 0)
    score_10 = float(s.get("score_10") or 0.0)
    b = s.get("buckets", {}) or {}
    b10_rate = float(b.get("10", 0) or 0) / max(1, n)
    ext_rate = float(s.get("extractable") or 0) / max(1, n)
    num_rate = float(s.get("numeric_pairs") or 0) / max(1, n)
    return (score_10, b10_rate, ext_rate, num_rate)

def is_reasoning_candidate(run):
    cfg = run.get("config", {})
    return cfg.get("train_style") not in {"answer_stub", "answer_only", "answer_focused"}

eligible = [
    run for run in runs
    if int(run.get("summary", {}).get("n") or 0) >= 500
    and float(run.get("summary", {}).get("score_10") or 0.0) > threshold
    and is_reasoning_candidate(run)
]

if not eligible and mode == "true":
    eligible = [run for run in runs if int(run.get("summary", {}).get("n") or 0) >= 500]

if not eligible:
    sys.exit(0)

best = sorted(eligible, key=score_key, reverse=True)[0]
cfg = best.get("config", {})
safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", best.get("name", "candidate"))
name = f"best_candidate_50k_1000_from_{safe_name}"

fields = [
    name,
    cfg.get("train_style", "equation_focused"),
    cfg.get("prompt_style", "answer_marker"),
    cfg.get("loss_style", "answer_line_only"),
    cfg.get("sampling_style", "balanced_type"),
    str(cfg.get("dedup_train_questions", False)).lower(),
    cfg.get("decoding_style", "short_greedy"),
    str(cfg.get("lr", 5e-5)),
    str(cfg.get("max_length", 384)),
    str(cfg.get("max_new_tokens", 64)),
    str(cfg.get("filter_non_numeric_gold", False)).lower(),
    deep_train,
    deep_valid,
    "1",
]
print("|".join(fields))
PY
  )"

  if [[ -n "$best_spec" ]]; then
    echo
    echo "===================================================================================================="
    echo "Running conditional larger candidate because a non-answer-stub 20k/500 run beat score_10 > $BEST_CANDIDATE_THRESHOLD"
    echo "Spec: $best_spec"
    echo "===================================================================================================="
    run_one "$best_spec"
  else
    echo
    echo "Skipping best_candidate_50k_1000: no non-answer-stub 20k/500 run beat score_10 > $BEST_CANDIDATE_THRESHOLD."
    echo "Set RUN_BEST_CANDIDATE=true to force a scaled candidate from the best 500-valid run."
  fi
fi

echo
echo "===================================================================================================="
echo "Final aggregate analysis"
echo "===================================================================================================="
python "$ANALYZE_PY" --root "$EXPERIMENTS_ROOT" \
  --out-json "$EXPERIMENTS_ROOT/experiment_summary.json" \
  --out-md "$EXPERIMENTS_ROOT/experiment_summary.md"

echo "Summary JSON: $EXPERIMENTS_ROOT/experiment_summary.json"
echo "Summary Markdown: $EXPERIMENTS_ROOT/experiment_summary.md"
