# %% [markdown]
# # Fine-tune Vietnamese GPT-2 for Math Word Problems
#
# Kaggle notebook script version.
#
# Candidate workflow:
# 1. Config
# 2. Load tokenizer/model
# 3. Load data
# 4. Light EDA
# 5. Dataset and collator
# 6. Evaluation utilities
# 7. Generation function
# 8. Optional baseline
# 9. Training
# 10. Validation generation
# 11. Validation evaluation
# 12. Error analysis
# 13. Phase 2 test prediction block
# 14. Technical report helper
#
# Default candidate:
# - Uses a frozen FDD/PoT checkpoint and inference-time candidate voting.
# - Keeps the model frozen: no trainer.train(), backward(), or optimizer step.
# - Keeps Python execution during inference disabled by default.
#
# Migration note:
# - Kaggle remains the official final run target. Keep Internet OFF there.
# - Vast.ai/Colab/local are only for experiments and can use path overrides or
#   optional kagglehub downloads.
# - Do not upload manually generated prediction files to Kaggle final runs.

# %%
import os
import sys
import json
import math
import time
import re
import random
import hashlib
import inspect
import ast
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

# %% [markdown]
# ## 1. Config

# %%
EXPERIMENT_MODE = "full_train"
# options:
# "eda_only"
# "small_debug"
# "small_compare"
# "full_train"
# "phase2_test"

TRAIN_STAGE = "sft"
# options:
# "sft"
# "grpo"
# "sft_then_grpo"
# "fdd_pot"

SFT_CKPT_DIR = ""

TRAIN_STYLE = "original_fdd_pot_restored"
# options:
# "original"
# "standardized"
# "short_solution"
# "answer_stub"
# "answer_only"
# "answer_focused"  # backward-compatible alias for answer_stub
# "equation_focused"
# "tagged_short_solution"
# "tagged_answer_only"
# "tagged_equation_focused"
# "tagged_equation_minimal"
# "preprocessed_target"
# "fdd_pot"
# "fdd_pot_compact"
# "fdd_lite_type_routing"
# "type_specific_equation_targets"
# "original_fdd_pot_restored"

PROMPT_STYLE = "pot_instruction"
# options:
# "plain"
# "instruction"
# "typed_instruction"
# "answer_marker"
# "typed_answer_marker"
# "direct_answer_marker"
# "direct_answer_only_marker"
# "pot_instruction"

LOSS_STYLE = "full_target_or_light_weighted"
# options:
# "full_target"
# "answer_line_only"
# "answer_weighted"
# "light_answer_weighted"
# "full_target_or_light_weighted"

SAMPLING_STYLE = "natural"
# options:
# "natural"
# "balanced_type"
# "gsm_heavy"
# "math_heavy"
# "easy_first"

TRAIN_TYPE_FILTER = ""
# options:
# ""    -> all train types
# "GSM" -> only train records whose type starts with GSM
# "MATH" -> only train records whose type starts with MATH

DEDUP_TRAIN_QUESTIONS = False

TAGGED_REASONING_MAX_CHARS = 700
TAGGED_REASONING_MAX_PARTS = 6

USE_TRAIN_RETRIEVAL_CONTEXT = False
N_RETRIEVED_EXAMPLES = 1

DECODING_STYLE = "greedy"
# options:
# "greedy"
# "beam2"
# "short_greedy"

SMOKE_TEST = False
FAST_DEV = True if SMOKE_TEST else False

MAX_TRAIN_SAMPLES = 3000 if SMOKE_TEST else None
MAX_VALID_SAMPLES = 200 if SMOKE_TEST else None
MAX_STEPS = 80 if SMOKE_TEST else -1

EPOCHS = 2
MAX_LENGTH = 384
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 7e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
SEED = 42

MAX_NEW_TOKENS = 64
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 4

FILTER_NO_ANSWER = False
FILTER_NON_NUMERIC_GOLD = True
RUN_BASELINE_FIRST = False
RUN_TRAIN = False
RUN_VALIDATION = True
RUN_TEST = True
POSTPROCESS_APPEND_LAST_NUMBER = True
GEN_BATCH_SIZE = 8

INFERENCE_VOTING = True
NUM_CANDIDATES = 3
ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING = True

TYPE_ROUTING = False
USE_LORA = False
USE_PEFT = False
RUN_PHASE2_CURRICULUM = False

BASE_RESPONSE_WEIGHT = 1.0
COMPUTATION_WEIGHT = 1.5
ANSWER_WEIGHT = 2.0

# Feedback-Driven Distillation / Program-of-Thought settings.
# The default uses PoT-like training targets but does not execute generated
# Python during inference.
FDD_EXECUTE_POT_AT_INFERENCE = False
FDD_MAX_EQUATION_STEPS = 3
FDD_PROGRAM_MAX_LINES = 24
FDD_KEEP_CODE_IN_OUTPUT = False

RUN_GRPO = False
GRPO_MAX_TRAIN_SAMPLES = 500
GRPO_ITERATIONS = 1
GRPO_ROLLOUT_BATCH_SIZE = 4
GRPO_NUM_GENERATIONS = 4
GRPO_BUFFER_SIZE = 128
GRPO_MINIBATCH_SIZE = 4
GRPO_EPOCHS_PER_BUFFER = 1
GRPO_LR = 1e-6
GRPO_CLIP_EPS = 0.2
GRPO_MAX_NEW_TOKENS = 128
GRPO_TEMPERATURE = 0.9
GRPO_TOP_P = 0.95
GRPO_TOP_K = 0
GRPO_GRAD_ACCUM = 1
GRPO_MAX_GRAD_NORM = 1.0
GRPO_CORRECTNESS_WEIGHT = 0.95
GRPO_FORMAT_WEIGHT = 0.05
GRPO_MIN_GROUP_REWARD_STD = 1e-8
GRPO_USE_THINK_ANSWER_TAGS = True
GRPO_USE_PARTIAL_NUMERIC_REWARD = False
GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER = True
GRPO_PREFLIGHT_ONLY = False
GRPO_EVAL_EVERY_ITERATION = True
GRPO_EARLY_STOP_ON_COLLAPSE = True

# Environment/path options.
# Official Kaggle runs should keep ALLOW_KAGGLEHUB_DOWNLOAD=False and use the
# attached Kaggle datasets. For Vast.ai/Colab experiments, set it to True after
# installing/configuring kagglehub, or set DATA_DIR_OVERRIDE/MODEL_DIR_OVERRIDE.
RUN_ENV = "auto"
# options: "auto", "kaggle", "vast", "local"

ALLOW_KAGGLEHUB_DOWNLOAD = False
DATASET_SLUG = "kimanh2002/dataset-math"
MODEL_DATASET_SLUG = "kimanh2002/nlphustgpt2-vietnamese"

DATA_DIR_OVERRIDE = os.environ.get("DATA_DIR_OVERRIDE", "").strip()
MODEL_DIR_OVERRIDE = os.environ.get("MODEL_DIR_OVERRIDE", "").strip()
WORKING_DIR_OVERRIDE = os.environ.get("WORKING_DIR", "").strip()
TRAIN_DATA_OVERRIDE = os.environ.get("TRAIN_DATA_OVERRIDE", "").strip()
PREPROCESSED_TARGET_FIELD = os.environ.get("PREPROCESSED_TARGET_FIELD", "target_direct").strip()


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int_or_none(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    raw = raw.strip()
    if raw.lower() in {"none", "null", "-1"}:
        return None
    return int(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw.strip()


# Automation hooks for Vast.ai/local experiment runners. Defaults preserve the
# notebook behavior when no environment variables are supplied.
EXPERIMENT_MODE = env_str("EXPERIMENT_MODE", EXPERIMENT_MODE)
TRAIN_STAGE = env_str("TRAIN_STAGE", TRAIN_STAGE)
SFT_CKPT_DIR = env_str("SFT_CKPT_DIR", SFT_CKPT_DIR)
TRAIN_STYLE = env_str("TRAIN_STYLE", TRAIN_STYLE)
PROMPT_STYLE = env_str("PROMPT_STYLE", PROMPT_STYLE)
LOSS_STYLE = env_str("LOSS_STYLE", LOSS_STYLE)
SAMPLING_STYLE = env_str("SAMPLING_STYLE", SAMPLING_STYLE)
TRAIN_TYPE_FILTER = env_str("TRAIN_TYPE_FILTER", TRAIN_TYPE_FILTER).upper()
DEDUP_TRAIN_QUESTIONS = env_bool("DEDUP_TRAIN_QUESTIONS", DEDUP_TRAIN_QUESTIONS)
TAGGED_REASONING_MAX_CHARS = env_int("TAGGED_REASONING_MAX_CHARS", TAGGED_REASONING_MAX_CHARS)
TAGGED_REASONING_MAX_PARTS = env_int("TAGGED_REASONING_MAX_PARTS", TAGGED_REASONING_MAX_PARTS)
USE_TRAIN_RETRIEVAL_CONTEXT = env_bool("USE_TRAIN_RETRIEVAL_CONTEXT", USE_TRAIN_RETRIEVAL_CONTEXT)
N_RETRIEVED_EXAMPLES = env_int("N_RETRIEVED_EXAMPLES", N_RETRIEVED_EXAMPLES)
DECODING_STYLE = env_str("DECODING_STYLE", DECODING_STYLE)
SMOKE_TEST = env_bool("SMOKE_TEST", SMOKE_TEST)
FAST_DEV = env_bool("FAST_DEV", FAST_DEV)
MAX_TRAIN_SAMPLES = env_int_or_none("MAX_TRAIN_SAMPLES", MAX_TRAIN_SAMPLES)
MAX_VALID_SAMPLES = env_int_or_none("MAX_VALID_SAMPLES", MAX_VALID_SAMPLES)
MAX_STEPS = env_int("MAX_STEPS", MAX_STEPS)
EPOCHS = env_int("EPOCHS", EPOCHS)
MAX_LENGTH = env_int("MAX_LENGTH", MAX_LENGTH)
PER_DEVICE_BATCH_SIZE = env_int("PER_DEVICE_BATCH_SIZE", PER_DEVICE_BATCH_SIZE)
GRAD_ACCUM = env_int("GRAD_ACCUM", GRAD_ACCUM)
LR = env_float("LR", LR)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", WEIGHT_DECAY)
WARMUP_RATIO = env_float("WARMUP_RATIO", WARMUP_RATIO)
SEED = env_int("SEED", SEED)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", MAX_NEW_TOKENS)
REPETITION_PENALTY = env_float("REPETITION_PENALTY", REPETITION_PENALTY)
NO_REPEAT_NGRAM_SIZE = env_int("NO_REPEAT_NGRAM_SIZE", NO_REPEAT_NGRAM_SIZE)
FILTER_NO_ANSWER = env_bool("FILTER_NO_ANSWER", FILTER_NO_ANSWER)
FILTER_NON_NUMERIC_GOLD = env_bool("FILTER_NON_NUMERIC_GOLD", FILTER_NON_NUMERIC_GOLD)
RUN_BASELINE_FIRST = env_bool("RUN_BASELINE_FIRST", RUN_BASELINE_FIRST)
RUN_TRAIN = env_bool("RUN_TRAIN", RUN_TRAIN)
RUN_VALIDATION = env_bool("RUN_VALIDATION", RUN_VALIDATION)
RUN_TEST = env_bool("RUN_TEST", RUN_TEST)
POSTPROCESS_APPEND_LAST_NUMBER = env_bool("POSTPROCESS_APPEND_LAST_NUMBER", POSTPROCESS_APPEND_LAST_NUMBER)
INFERENCE_VOTING = env_bool("INFERENCE_VOTING", INFERENCE_VOTING)
NUM_CANDIDATES = env_int("NUM_CANDIDATES", NUM_CANDIDATES)
ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING = env_bool(
    "ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING",
    ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING,
)
RUN_GRPO = env_bool("RUN_GRPO", RUN_GRPO)
GRPO_MAX_TRAIN_SAMPLES = env_int("GRPO_MAX_TRAIN_SAMPLES", GRPO_MAX_TRAIN_SAMPLES)
GRPO_ITERATIONS = env_int("GRPO_ITERATIONS", GRPO_ITERATIONS)
GRPO_ROLLOUT_BATCH_SIZE = env_int("GRPO_ROLLOUT_BATCH_SIZE", GRPO_ROLLOUT_BATCH_SIZE)
GRPO_NUM_GENERATIONS = env_int("GRPO_NUM_GENERATIONS", GRPO_NUM_GENERATIONS)
GRPO_BUFFER_SIZE = env_int("GRPO_BUFFER_SIZE", GRPO_BUFFER_SIZE)
GRPO_MINIBATCH_SIZE = env_int("GRPO_MINIBATCH_SIZE", GRPO_MINIBATCH_SIZE)
GRPO_EPOCHS_PER_BUFFER = env_int("GRPO_EPOCHS_PER_BUFFER", GRPO_EPOCHS_PER_BUFFER)
GRPO_LR = env_float("GRPO_LR", GRPO_LR)
GRPO_CLIP_EPS = env_float("GRPO_CLIP_EPS", GRPO_CLIP_EPS)
GRPO_MAX_NEW_TOKENS = env_int("GRPO_MAX_NEW_TOKENS", GRPO_MAX_NEW_TOKENS)
GRPO_TEMPERATURE = env_float("GRPO_TEMPERATURE", GRPO_TEMPERATURE)
GRPO_TOP_P = env_float("GRPO_TOP_P", GRPO_TOP_P)
GRPO_TOP_K = env_int("GRPO_TOP_K", GRPO_TOP_K)
GRPO_GRAD_ACCUM = env_int("GRPO_GRAD_ACCUM", GRPO_GRAD_ACCUM)
GRPO_MAX_GRAD_NORM = env_float("GRPO_MAX_GRAD_NORM", GRPO_MAX_GRAD_NORM)
GRPO_CORRECTNESS_WEIGHT = env_float("GRPO_CORRECTNESS_WEIGHT", GRPO_CORRECTNESS_WEIGHT)
GRPO_FORMAT_WEIGHT = env_float("GRPO_FORMAT_WEIGHT", GRPO_FORMAT_WEIGHT)
GRPO_MIN_GROUP_REWARD_STD = env_float("GRPO_MIN_GROUP_REWARD_STD", GRPO_MIN_GROUP_REWARD_STD)
GRPO_USE_THINK_ANSWER_TAGS = env_bool("GRPO_USE_THINK_ANSWER_TAGS", GRPO_USE_THINK_ANSWER_TAGS)
GRPO_USE_PARTIAL_NUMERIC_REWARD = env_bool("GRPO_USE_PARTIAL_NUMERIC_REWARD", GRPO_USE_PARTIAL_NUMERIC_REWARD)
GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER = env_bool(
    "GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER",
    GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER,
)
GRPO_PREFLIGHT_ONLY = env_bool("GRPO_PREFLIGHT_ONLY", GRPO_PREFLIGHT_ONLY)
GRPO_EVAL_EVERY_ITERATION = env_bool("GRPO_EVAL_EVERY_ITERATION", GRPO_EVAL_EVERY_ITERATION)
GRPO_EARLY_STOP_ON_COLLAPSE = env_bool("GRPO_EARLY_STOP_ON_COLLAPSE", GRPO_EARLY_STOP_ON_COLLAPSE)
RUN_ENV = env_str("RUN_ENV", RUN_ENV)
ALLOW_KAGGLEHUB_DOWNLOAD = env_bool("ALLOW_KAGGLEHUB_DOWNLOAD", ALLOW_KAGGLEHUB_DOWNLOAD)
GEN_BATCH_SIZE = env_int("GEN_BATCH_SIZE", GEN_BATCH_SIZE)
TRAIN_DATA_OVERRIDE = env_str("TRAIN_DATA_OVERRIDE", TRAIN_DATA_OVERRIDE)
PREPROCESSED_TARGET_FIELD = env_str("PREPROCESSED_TARGET_FIELD", PREPROCESSED_TARGET_FIELD)
TYPE_ROUTING = env_bool("TYPE_ROUTING", TYPE_ROUTING)
USE_LORA = env_bool("USE_LORA", USE_LORA)
USE_PEFT = env_bool("USE_PEFT", USE_PEFT)
RUN_PHASE2_CURRICULUM = env_bool("RUN_PHASE2_CURRICULUM", RUN_PHASE2_CURRICULUM)
BASE_RESPONSE_WEIGHT = env_float("BASE_RESPONSE_WEIGHT", BASE_RESPONSE_WEIGHT)
COMPUTATION_WEIGHT = env_float("COMPUTATION_WEIGHT", COMPUTATION_WEIGHT)
ANSWER_WEIGHT = env_float("ANSWER_WEIGHT", ANSWER_WEIGHT)
FDD_EXECUTE_POT_AT_INFERENCE = env_bool("FDD_EXECUTE_POT_AT_INFERENCE", FDD_EXECUTE_POT_AT_INFERENCE)
FDD_MAX_EQUATION_STEPS = env_int("FDD_MAX_EQUATION_STEPS", FDD_MAX_EQUATION_STEPS)
FDD_PROGRAM_MAX_LINES = env_int("FDD_PROGRAM_MAX_LINES", FDD_PROGRAM_MAX_LINES)
FDD_KEEP_CODE_IN_OUTPUT = env_bool("FDD_KEEP_CODE_IN_OUTPUT", FDD_KEEP_CODE_IN_OUTPUT)

if TRAIN_STAGE not in {"sft", "grpo", "sft_then_grpo", "fdd_pot"}:
    raise ValueError(f"Unknown TRAIN_STAGE: {TRAIN_STAGE}")

if TRAIN_TYPE_FILTER not in {"", "GSM", "MATH"}:
    raise ValueError(f"Unknown TRAIN_TYPE_FILTER: {TRAIN_TYPE_FILTER}")

if TRAIN_STAGE in {"grpo", "sft_then_grpo"}:
    RUN_GRPO = True

# Mode convenience switches. Keep the main values explicit above so each Kaggle
# commit can change one or two variables and compare validation behavior.
if EXPERIMENT_MODE == "eda_only":
    RUN_TRAIN = False
    RUN_VALIDATION = False
    RUN_TEST = False
    RUN_GRPO = False
elif EXPERIMENT_MODE == "phase2_test":
    RUN_TRAIN = False
    RUN_VALIDATION = False
    RUN_TEST = True
    RUN_GRPO = False

if INFERENCE_VOTING:
    RUN_TRAIN = False
    RUN_GRPO = False

SAFE_EOS_ID = 50256

def detect_run_env() -> str:
    if RUN_ENV != "auto":
        return RUN_ENV
    if Path("/kaggle/input").exists():
        return "kaggle"
    if Path("/workspace").exists() or any(k.startswith("VAST_") for k in os.environ):
        return "vast"
    return "local"


ACTIVE_RUN_ENV = detect_run_env()


def default_working_dir() -> Path:
    if WORKING_DIR_OVERRIDE:
        return Path(WORKING_DIR_OVERRIDE)
    if ACTIVE_RUN_ENV == "kaggle":
        return Path("/kaggle/working")
    if ACTIVE_RUN_ENV == "vast" and Path("/workspace").exists():
        return Path("/workspace/kaggle_working")
    return Path("./working")


WORKING_DIR = default_working_dir()
WORKING_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = WORKING_DIR / "gpt2_math_ckpt"
GRPO_OUTPUT_DIR = WORKING_DIR / "gpt2_math_grpo_ckpt"

VALID_OUTPUT_PATH = WORKING_DIR / "valid_output.json"
VALID_REPORT_PATH = WORKING_DIR / "valid_report.json"

BASELINE_OUTPUT_PATH = WORKING_DIR / "baseline_valid_output.json"
BASELINE_REPORT_PATH = WORKING_DIR / "baseline_valid_report.json"

TEST_OUTPUT_PATH = WORKING_DIR / "test_predictions.json"

EXPERIMENT_REPORT_PATH = WORKING_DIR / "experiment_report.json"
ERROR_ANALYSIS_PATH = WORKING_DIR / "error_analysis.json"
GRPO_PREFLIGHT_PATH = WORKING_DIR / "grpo_preflight_diagnostics.json"
GREEDY_VALID_OUTPUT_PATH = WORKING_DIR / "greedy_valid_output.json"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

CONFIG = {
    "experiment_mode": EXPERIMENT_MODE,
    "train_stage": TRAIN_STAGE,
    "sft_ckpt_dir": SFT_CKPT_DIR,
    "train_style": TRAIN_STYLE,
    "prompt_style": PROMPT_STYLE,
    "loss_style": LOSS_STYLE,
    "sampling_style": SAMPLING_STYLE,
    "train_type_filter": TRAIN_TYPE_FILTER,
    "dedup_train_questions": DEDUP_TRAIN_QUESTIONS,
    "tagged_reasoning_max_chars": TAGGED_REASONING_MAX_CHARS,
    "tagged_reasoning_max_parts": TAGGED_REASONING_MAX_PARTS,
    "use_train_retrieval_context": USE_TRAIN_RETRIEVAL_CONTEXT,
    "n_retrieved_examples": N_RETRIEVED_EXAMPLES,
    "decoding_style": DECODING_STYLE,
    "smoke_test": SMOKE_TEST,
    "fast_dev": FAST_DEV,
    "max_train_samples": MAX_TRAIN_SAMPLES,
    "max_valid_samples": MAX_VALID_SAMPLES,
    "max_steps": MAX_STEPS,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "seed": SEED,
    "max_new_tokens": MAX_NEW_TOKENS,
    "repetition_penalty": REPETITION_PENALTY,
    "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    "filter_no_answer": FILTER_NO_ANSWER,
    "filter_non_numeric_gold": FILTER_NON_NUMERIC_GOLD,
    "run_baseline_first": RUN_BASELINE_FIRST,
    "run_train": RUN_TRAIN,
    "run_validation": RUN_VALIDATION,
    "run_test": RUN_TEST,
    "postprocess_append_last_number": POSTPROCESS_APPEND_LAST_NUMBER,
    "inference_voting": INFERENCE_VOTING,
    "num_candidates": NUM_CANDIDATES,
    "allow_base_model_fallback_for_voting": ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING,
    "run_grpo": RUN_GRPO,
    "grpo_output_dir": str(GRPO_OUTPUT_DIR),
    "grpo_max_train_samples": GRPO_MAX_TRAIN_SAMPLES,
    "grpo_iterations": GRPO_ITERATIONS,
    "grpo_rollout_batch_size": GRPO_ROLLOUT_BATCH_SIZE,
    "grpo_num_generations": GRPO_NUM_GENERATIONS,
    "grpo_buffer_size": GRPO_BUFFER_SIZE,
    "grpo_minibatch_size": GRPO_MINIBATCH_SIZE,
    "grpo_epochs_per_buffer": GRPO_EPOCHS_PER_BUFFER,
    "grpo_lr": GRPO_LR,
    "grpo_clip_eps": GRPO_CLIP_EPS,
    "grpo_max_new_tokens": GRPO_MAX_NEW_TOKENS,
    "grpo_temperature": GRPO_TEMPERATURE,
    "grpo_top_p": GRPO_TOP_P,
    "grpo_top_k": GRPO_TOP_K,
    "grpo_grad_accum": GRPO_GRAD_ACCUM,
    "grpo_max_grad_norm": GRPO_MAX_GRAD_NORM,
    "grpo_correctness_weight": GRPO_CORRECTNESS_WEIGHT,
    "grpo_format_weight": GRPO_FORMAT_WEIGHT,
    "grpo_min_group_reward_std": GRPO_MIN_GROUP_REWARD_STD,
    "grpo_use_think_answer_tags": GRPO_USE_THINK_ANSWER_TAGS,
    "grpo_use_partial_numeric_reward": GRPO_USE_PARTIAL_NUMERIC_REWARD,
    "grpo_gate_format_reward_on_numeric_answer": GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER,
    "grpo_preflight_only": GRPO_PREFLIGHT_ONLY,
    "grpo_eval_every_iteration": GRPO_EVAL_EVERY_ITERATION,
    "grpo_early_stop_on_collapse": GRPO_EARLY_STOP_ON_COLLAPSE,
    "run_env": ACTIVE_RUN_ENV,
    "allow_kagglehub_download": ALLOW_KAGGLEHUB_DOWNLOAD,
    "data_dir_override": DATA_DIR_OVERRIDE,
    "model_dir_override": MODEL_DIR_OVERRIDE,
    "train_data_override": TRAIN_DATA_OVERRIDE,
    "preprocessed_target_field": PREPROCESSED_TARGET_FIELD,
    "type_routing": TYPE_ROUTING,
    "use_lora": USE_LORA,
    "use_peft": USE_PEFT,
    "phase2_curriculum": RUN_PHASE2_CURRICULUM,
    "base_response_weight": BASE_RESPONSE_WEIGHT,
    "computation_weight": COMPUTATION_WEIGHT,
    "answer_weight": ANSWER_WEIGHT,
    "fdd_execute_pot_at_inference": FDD_EXECUTE_POT_AT_INFERENCE,
    "fdd_max_equation_steps": FDD_MAX_EQUATION_STEPS,
    "fdd_program_max_lines": FDD_PROGRAM_MAX_LINES,
    "fdd_keep_code_in_output": FDD_KEEP_CODE_IN_OUTPUT,
    "working_dir": str(WORKING_DIR),
    "safe_eos_id": SAFE_EOS_ID,
}

print(json.dumps(CONFIG, ensure_ascii=False, indent=2))

# %% [markdown]
# ### Vast.ai / Local Migration Workflow
#
# This notebook uses the same training and evaluation code across environments.
#
# **Official Kaggle run**
# - Attach `kimanh2002/dataset-math` and `kimanh2002/nlphustgpt2-vietnamese`.
# - Keep `ALLOW_KAGGLEHUB_DOWNLOAD = False`.
# - Keep Internet OFF.
# - Outputs are written to `/kaggle/working`.
#
# **Vast.ai experiments**
# - Use Vast for debugging, ablations, and longer experiments only.
# - Option A: set `ALLOW_KAGGLEHUB_DOWNLOAD = True` and configure Kaggle credentials for `kagglehub`.
# - Option B: manually place folders on the instance and set environment variables:
#   - `DATA_DIR_OVERRIDE=/path/to/dataset-math`
#   - `MODEL_DIR_OVERRIDE=/path/to/nlphustgpt2-vietnamese`
#   - optional `WORKING_DIR=/workspace/kaggle_working`
# - Do not copy Vast-generated predictions into the final Kaggle submission.
#
# **Colab/local experiments**
# - Use the same override variables or `ALLOW_KAGGLEHUB_DOWNLOAD=True`.
# - Treat results as development evidence; final scoring must still come from a Kaggle Save Version / Run All.

# %% [markdown]
# ## 2. Paths, Tokenizer/Model Check, and Data Loading

# %%
def first_existing(*paths: str | Path) -> Path:
    for p in map(Path, paths):
        if p.exists():
            return p
    raise FileNotFoundError("Không tìm thấy path nào: " + " | ".join(map(str, paths)))


def first_existing_or_none(*paths: str | Path) -> Optional[Path]:
    for p in map(Path, paths):
        if p and p.exists():
            return p
    return None


def kagglehub_download(slug: str, label: str) -> Optional[Path]:
    if not ALLOW_KAGGLEHUB_DOWNLOAD:
        return None
    if ACTIVE_RUN_ENV == "kaggle":
        raise RuntimeError(
            "ALLOW_KAGGLEHUB_DOWNLOAD=True is not allowed for the official Kaggle offline run. "
            "Attach the dataset/model as Kaggle inputs instead."
        )
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. On Vast.ai/Colab experiments, install it first or "
            "set DATA_DIR_OVERRIDE and MODEL_DIR_OVERRIDE to existing local folders."
        ) from exc

    path = Path(kagglehub.dataset_download(slug))
    print(f"{label} downloaded/resolved by kagglehub:", path)
    return path


def resolve_data_dir() -> Path:
    candidates = [
        DATA_DIR_OVERRIDE,
        "/kaggle/input/datasets/kimanh2002/dataset-math",
        "/kaggle/input/dataset-math",
        "/workspace/input/dataset-math",
        "/workspace/dataset-math",
        "./input/dataset-math",
        "./dataset-math",
    ]
    found = first_existing_or_none(*[x for x in candidates if x])
    if found is not None:
        return found

    downloaded = kagglehub_download(DATASET_SLUG, "Dataset")
    if downloaded is not None:
        return downloaded

    raise FileNotFoundError(
        "Không tìm thấy dataset. Trên Kaggle, attach dataset-math. "
        "Trên Vast.ai/Colab, set ALLOW_KAGGLEHUB_DOWNLOAD=True hoặc DATA_DIR_OVERRIDE."
    )


def resolve_model_dir() -> Path:
    candidates = [
        MODEL_DIR_OVERRIDE,
        "/kaggle/input/datasets/kimanh2002/nlphustgpt2-vietnamese",
        "/kaggle/input/nlphustgpt2-vietnamese",
        "/kaggle/input/nlphustgpt2-vietnamese/gpt2-vietnamese",
        "/workspace/input/nlphustgpt2-vietnamese",
        "/workspace/nlphustgpt2-vietnamese",
        "./input/nlphustgpt2-vietnamese",
        "./nlphustgpt2-vietnamese",
        "./gpt2-vietnamese",
    ]
    found = first_existing_or_none(*[x for x in candidates if x])
    if found is not None:
        return found

    downloaded = kagglehub_download(MODEL_DATASET_SLUG, "Base model")
    if downloaded is not None:
        nested = first_existing_or_none(
            downloaded,
            downloaded / "gpt2-vietnamese",
            downloaded / "nlphustgpt2-vietnamese",
        )
        if nested is not None:
            return nested

    raise FileNotFoundError(
        "Không tìm thấy local GPT-2 model folder. Trên Kaggle, attach nlphustgpt2-vietnamese. "
        "Trên Vast.ai/Colab, set ALLOW_KAGGLEHUB_DOWNLOAD=True hoặc MODEL_DIR_OVERRIDE."
    )


DATA_DIR = resolve_data_dir()

MODEL_NAME = str(resolve_model_dir())

TRAIN_FILE = DATA_DIR / "train.json"
VALID_FILE = DATA_DIR / "valid.json"
TEST_FILE = DATA_DIR / "test.json"
TRAIN_SOURCE_FILE = Path(TRAIN_DATA_OVERRIDE) if TRAIN_DATA_OVERRIDE else TRAIN_FILE

print("TRAIN_FILE:", TRAIN_FILE)
print("TRAIN_SOURCE_FILE:", TRAIN_SOURCE_FILE, "| override:", bool(TRAIN_DATA_OVERRIDE))
print("VALID_FILE:", VALID_FILE)
print("TEST_FILE:", TEST_FILE, "| exists:", TEST_FILE.exists())
print("MODEL_NAME:", MODEL_NAME)
print("ACTIVE_RUN_ENV:", ACTIVE_RUN_ENV)
print("WORKING_DIR:", WORKING_DIR)


def configure_tokenizer(tok: Any) -> Any:
    tok.pad_token_id = SAFE_EOS_ID
    tok.eos_token_id = SAFE_EOS_ID
    return tok


tokenizer = configure_tokenizer(
    AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
)
print("Tokenizer:", type(tokenizer), "| vocab size:", len(tokenizer))

SPECIAL_TOKEN_ARTIFACTS: list[str] = []


def tokenizer_sanity_check(tok: Any) -> None:
    raw = tok.decode([SAFE_EOS_ID], skip_special_tokens=False)
    skipped = tok.decode([SAFE_EOS_ID], skip_special_tokens=True)
    print("SAFE_EOS_ID:", SAFE_EOS_ID)
    print("decode SAFE_EOS_ID raw:", repr(raw))
    print("decode SAFE_EOS_ID skip:", repr(skipped))
    print("tokenizer.eos_token:", repr(tok.eos_token))
    print("tokenizer.eos_token_id:", tok.eos_token_id)
    print("tokenizer.pad_token:", repr(tok.pad_token))
    print("tokenizer.pad_token_id:", tok.pad_token_id)

    if raw:
        SPECIAL_TOKEN_ARTIFACTS.append(raw)
    for val in (tok.eos_token, tok.pad_token):
        if isinstance(val, str) and val:
            SPECIAL_TOKEN_ARTIFACTS.append(val)

    # Keep unique artifacts, longest first, so "<eos>" is stripped before "eos".
    uniq = sorted(set(SPECIAL_TOKEN_ARTIFACTS), key=len, reverse=True)
    SPECIAL_TOKEN_ARTIFACTS.clear()
    SPECIAL_TOKEN_ARTIFACTS.extend(uniq)
    print("special token artifacts stripped from debug/output tails:", [repr(x) for x in SPECIAL_TOKEN_ARTIFACTS])


tokenizer_sanity_check(tokenizer)

# Quick offline model-load check. Training/generation reloads the model later.
model_check = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
model_check.config.pad_token_id = SAFE_EOS_ID
model_check.config.eos_token_id = SAFE_EOS_ID
print("Model:", type(model_check), "| config vocab_size:", model_check.config.vocab_size)
del model_check
torch.cuda.empty_cache()


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"{p} must contain a JSON array or JSONL records.")
    return data


train_records_all = load_records(TRAIN_SOURCE_FILE)
valid_records_all = load_records(VALID_FILE)

train_records = train_records_all
valid_records = valid_records_all

print("raw train:", len(train_records_all))
print("raw valid:", len(valid_records_all))
print("available keys:", sorted(train_records_all[0].keys()) if train_records_all else [])
if TRAIN_DATA_OVERRIDE:
    print(
        "Using TRAIN_DATA_OVERRIDE for SFT train records only. "
        "Validation and test still come from DATA_DIR valid.json/test.json."
    )
print("sample record:")
print(json.dumps(train_records_all[0], ensure_ascii=False, indent=2)[:1600] if train_records_all else "[]")

# %% [markdown]
# ## Shared Helpers for Reports, Prompts, Targets, and Answer Extraction

# %%
def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json_if_exists(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_experiment_report(**sections: Any) -> dict:
    report = load_json_if_exists(EXPERIMENT_REPORT_PATH)
    report.setdefault("config", CONFIG)
    report.setdefault("paths", {
        "data_dir": str(DATA_DIR),
        "train_source_file": str(TRAIN_SOURCE_FILE),
        "model_name": str(MODEL_NAME),
        "output_dir": str(OUTPUT_DIR),
        "grpo_output_dir": str(GRPO_OUTPUT_DIR),
        "valid_output_path": str(VALID_OUTPUT_PATH),
        "valid_report_path": str(VALID_REPORT_PATH),
        "test_output_path": str(TEST_OUTPUT_PATH),
        "grpo_preflight_path": str(GRPO_PREFLIGHT_PATH),
    })
    report.update(sections)
    save_json(report, EXPERIMENT_REPORT_PATH)
    return report


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(
    dir_path: str | Path,
    suffixes: tuple[str, ...] = (".bin", ".safetensors", ".json", ".txt", ".model"),
) -> str:
    root = Path(dir_path)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix in suffixes):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(sha256_file(p).encode() + b"\0")
    return h.hexdigest()


ANSWER_MARKER_PATTERNS = [
    r"####\s*đáp\s*án\s*là\s*[:：]?",
    r"Đáp\s*án\s*là\s*[:：]?",
    r"Đáp\s*án\s*[:：]",
    r"Câu\s*trả\s*lời\s*là\s*[:：]?",
    r"Câu\s*trả\s*lời\s*[:：]",
    r"Đáp\s*số\s*là\s*[:：]?",
    r"Đáp\s*số\s*[:：]",
    r"The\s+answer\s+is\s*[:：]?",
    r"####\s*",
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
SAFE_NS = {"sqrt": math.sqrt, "pi": math.pi}
ANSWER_ANCHOR = "####đáp án là:"

_SIMPLE: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
    "muời": 10,
}


def normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_trailing_special_artifacts(text: str) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        out = out.rstrip()
        for artifact in SPECIAL_TOKEN_ARTIFACTS:
            if artifact and out.endswith(artifact):
                out = out[: -len(artifact)]
                changed = True
    return out.rstrip()


def cleanup_answer_string(text: str) -> str:
    ans = normalize_space(text)

    # Cut duplicate final markers if a generated line repeats them.
    cut_positions = []
    for pat in ANSWER_MARKER_PATTERNS:
        m = re.search(pat, ans, flags=re.IGNORECASE)
        if m and m.start() > 0:
            cut_positions.append(m.start())
    if cut_positions:
        ans = ans[: min(cut_positions)].strip()

    # Keep the first answer line only.
    ans = ans.split("\n")[0].strip()
    ans = ans.strip(" \t:：")
    ans = ans.strip("\"'“”‘’`")
    ans = strip_trailing_special_artifacts(ans)
    ans = ans.rstrip(" .。;；")
    ans = ans.strip("\"'“”‘’`")
    ans = strip_trailing_special_artifacts(ans)
    # The GPT-2 EOS artifact can decode as "hue"; strip repeated tails without
    # touching symbolic answers such as 50\sqrt{10}.
    ans = re.sub(r"(?<=\d)(?:hue)+$", "", ans, flags=re.IGNORECASE)
    return ans


def count_answer_anchors(text: str) -> int:
    return len(re.findall(r"####\s*đáp\s*án\s*là\s*:", text or "", flags=re.IGNORECASE))


def extract_answer(text: Optional[str]) -> Optional[str]:
    """Extract a final answer string from Vietnamese/GSM/LaTeX answer markers."""
    if not text:
        return None

    text = str(text)
    matches: list[tuple[int, int]] = []
    for pat in ANSWER_MARKER_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end()))

    if matches:
        # Last marker is usually the final answer when generations repeat.
        # If generic "####" and specific "####đáp án là:" both start at the
        # same location, prefer the longer/specific marker.
        _, end = sorted(matches, key=lambda x: (x[0], x[1]))[-1]
        ans = cleanup_answer_string(text[end:])
        if ans:
            return ans

    boxes = BOXED_RE.findall(text)
    if boxes:
        ans = cleanup_answer_string(boxes[-1])
        return ans or None

    return None


def remove_final_answer_tail(text: str) -> str:
    """Remove final answer marker tail while keeping earlier reasoning."""
    text = normalize_space(text)
    matches: list[tuple[int, int]] = []
    for pat in ANSWER_MARKER_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            matches.append((m.start(), m.end()))
    if not matches:
        return text

    # Prefer the first marker in the final answer cluster near the end.
    near_tail = [m for m in matches if len(text) - m[0] <= 700]
    cut = min(m[0] for m in near_tail) if near_tail else sorted(matches)[-1][0]
    return normalize_space(text[:cut])


def format_retrieval_context(examples: Optional[list[dict]]) -> str:
    if not examples:
        return ""

    lines = [
        "Ví dụ tương tự từ train.json (chỉ dùng nếu được phép):",
    ]
    for k, ex in enumerate(examples[:N_RETRIEVED_EXAMPLES], 1):
        q = normalize_space(str(ex.get("query_vi", "")))
        a = normalize_space(str(ex.get("response_vi", "")))
        lines.append(f"Ví dụ {k} - Câu hỏi: {q}")
        lines.append(f"Ví dụ {k} - Lời giải: {a}")
    lines.append("")
    return "\n".join(lines)


def get_query_text(rec: dict) -> str:
    """Return the best available question field for raw or preprocessed records."""
    for key in ("query_vi_clean", "query_vi_original", "query_vi"):
        val = normalize_space(str(rec.get(key, "") or ""))
        if val:
            return val
    return ""


def build_prompt(
    rec: dict,
    prompt_style: str = PROMPT_STYLE,
    retrieved_examples: Optional[list[dict]] = None,
) -> str:
    query = get_query_text(rec)
    problem_type = str(rec.get("type", "") or "unknown").strip()
    prefix = format_retrieval_context(retrieved_examples)

    if prompt_style == "plain":
        return prefix + f"Câu hỏi: {query}\nLời giải:"
    if prompt_style == "instruction":
        return prefix + (
            'Hãy giải bài toán ngắn gọn và kết thúc bằng dòng "Đáp án là: <số>".\n'
            f"Câu hỏi: {query}\n"
            "Lời giải:"
        )
    if prompt_style == "typed_instruction":
        return prefix + (
            'Hãy giải bài toán ngắn gọn và kết thúc bằng dòng "Đáp án là: <số>".\n'
            f"Dạng bài: {problem_type}\n"
            f"Câu hỏi: {query}\n"
            "Lời giải:"
        )
    if prompt_style == "answer_marker":
        return prefix + (
            "Hãy giải bài toán ngắn gọn. Dòng cuối phải bắt đầu bằng: ####đáp án là:\n"
            f"Câu hỏi: {query}\n"
            "Lời giải:"
        )
    if prompt_style == "typed_answer_marker":
        return prefix + (
            "Hãy giải bài toán ngắn gọn. Dòng cuối phải bắt đầu bằng: ####đáp án là:\n"
            f"Dạng bài: {problem_type}\n"
            f"Câu hỏi: {query}\n"
            "Lời giải:"
        )
    if prompt_style == "direct_answer_marker":
        return prefix + (
            'Hãy giải bài toán bằng vài phép tính ngắn. Dòng cuối phải là "####đáp án là: <số>".\n'
            f"Câu hỏi: {query}\n"
            "Lời giải ngắn:"
        )
    if prompt_style == "direct_answer_only_marker":
        return prefix + (
            "Hãy giải bài toán và chỉ đưa ra đáp án cuối cùng.\n"
            f"Câu hỏi: {query}\n"
            "Đáp án là:"
        )
    if prompt_style == "pot_instruction":
        return prefix + (
            "Hãy giải bài toán bằng vài dòng phương trình ngắn. "
            "Ưu tiên phép tính rõ ràng và kết thúc bằng dòng ####đáp án là: <số>.\n"
            f"Câu hỏi: {query}\n"
            "Lời giải ngắn:"
        )
    raise ValueError(f"Unknown PROMPT_STYLE: {prompt_style}")


def build_grpo_prompt(rec: dict) -> str:
    query = get_query_text(rec)
    problem_type = normalize_space(str(rec.get("type", "") or ""))
    type_line = f"Dạng bài: {problem_type}\n" if problem_type else ""
    return (
        "Hãy giải bài toán. Trước tiên viết phần suy luận trong thẻ <think>...</think>, "
        "sau đó viết đáp án cuối cùng trong thẻ <answer>...</answer>.\n"
        "Chỉ nội dung trong <answer> sẽ được chấm điểm.\n\n"
        f"{type_line}"
        f"Câu hỏi: {query}\n\n"
        "Bài giải:\n"
    )


def uses_grpo_tagged_target(train_style: str) -> bool:
    return train_style in {
        "tagged_short_solution",
        "tagged_answer_only",
        "tagged_equation_focused",
        "tagged_equation_minimal",
    }


def build_inference_prompt(
    rec: dict,
    retrieved_examples: Optional[list[dict]] = None,
) -> str:
    if GRPO_USE_THINK_ANSWER_TAGS and (
        TRAIN_STAGE in {"grpo", "sft_then_grpo"} or uses_grpo_tagged_target(TRAIN_STYLE)
    ):
        return build_grpo_prompt(rec)
    return build_prompt(rec, PROMPT_STYLE, retrieved_examples)


def split_sentences_vietnamese(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def shorten_reasoning(text: str, max_chars: int = 450, max_sentences: int = 4) -> str:
    text = remove_final_answer_tail(text)
    sentences = split_sentences_vietnamese(text)
    if sentences:
        short = " ".join(sentences[:max_sentences]).strip()
    else:
        short = text

    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].strip()
    return short


def equation_focused_reasoning(text: str, max_chars: int = 700, max_parts: int = 6) -> str:
    """Keep compact reasoning pieces that contain equations, operators, or numbers."""
    body = remove_final_answer_tail(text)
    if not body:
        return ""

    parts = []
    for raw in re.split(r"(?<=[.!?。])\s+|\n+", body):
        part = raw.strip()
        if not part:
            continue
        has_numeric_signal = bool(re.search(r"\d|=|\\frac|\\sqrt|\\boxed|\+|\-|\*|/|%", part))
        if has_numeric_signal:
            parts.append(part)

    if not parts:
        parts = split_sentences_vietnamese(body)[:3]

    # Preserve early setup and late calculation, but avoid long repetitive targets.
    if len(parts) > max_parts:
        keep_head = max_parts // 2
        keep_tail = max_parts - keep_head
        parts = parts[:keep_head] + parts[-keep_tail:]

    short = " ".join(parts).strip()
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0].strip()
    return short


EQUATION_OPERATOR_RE = re.compile(r"(=|\+|\-|\*|/|×|%|\\frac)")


def minimal_equation_reasoning(text: str, max_chars: int = 300, max_parts: int = 3) -> str:
    """Keep only compact numeric/equation fragments near the final answer."""
    max_chars = min(max(80, max_chars), 300)
    max_parts = min(max(1, max_parts), 3)
    source = equation_focused_reasoning(
        text,
        max_chars=max(700, max_chars * 3),
        max_parts=max(8, max_parts * 4),
    )
    if not source:
        return ""

    raw_parts = re.split(
        r"(?<=[.!?。])\s+|\n+|;|；|\s+\bvậy\b\s+|\s+\bdo đó\b\s+|\s+\bkhi đó\b\s+",
        source,
        flags=re.IGNORECASE,
    )
    candidates = []
    for idx, raw in enumerate(raw_parts):
        part = normalize_space(raw).strip(" .。,:;")
        if not part or not re.search(r"\d", part):
            continue
        if not EQUATION_OPERATOR_RE.search(part):
            continue

        # Drop front-loaded prose around the calculation. Keep the local clause
        # containing the operator rather than the whole explanatory sentence.
        clauses = re.split(r",|，|\s+\bvà\b\s+|\s+\bnên\b\s+", part, flags=re.IGNORECASE)
        useful = []
        for clause in clauses:
            clause = normalize_space(clause).strip(" .。,:;")
            if re.search(r"\d", clause) and EQUATION_OPERATOR_RE.search(clause):
                useful.append(clause)
        if useful:
            part = min(useful, key=len)

        if len(part) > 140:
            op = EQUATION_OPERATOR_RE.search(part)
            if op:
                left = max(0, op.start() - 55)
                right = min(len(part), op.end() + 85)
                part = part[left:right].strip(" .。,:;")

        candidates.append((idx, len(part), part))

    if not candidates:
        return ""

    # Prefer snippets closest to the final answer, then keep them in original order.
    chosen = sorted(candidates, key=lambda x: (-x[0], x[1]))[:max_parts]
    chosen = sorted(chosen, key=lambda x: x[0])
    parts = []
    seen = set()
    for _, _, part in chosen:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(part)

    out = "\n".join(parts).strip()
    if len(out) > max_chars:
        clipped = []
        total = 0
        for part in parts:
            add_len = len(part) + (1 if clipped else 0)
            if total + add_len > max_chars:
                break
            clipped.append(part)
            total += add_len
        out = "\n".join(clipped).strip()
    return out


def numeric_literal(value: float, original: str) -> str:
    cleaned = cleanup_answer_string(original)
    if re.fullmatch(r"-?\d+", cleaned):
        return cleaned
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return repr(float(value))


def expression_to_python(expr: str) -> Optional[str]:
    """Convert a compact arithmetic/LaTeX expression into safe Python syntax."""
    t = normalize_space(expr)
    if not t:
        return None
    t = t.replace("$", "")
    for _ in range(3):
        new = re.sub(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r"(\1)", t)
        if new == t:
            break
        t = new
    t = re.sub(r"\\text\{[^}]*\}", "", t)
    t = re.sub(r"\\mathrm\{[^}]*\}", "", t)
    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right"):
        t = t.replace(token, "")
    for token in ("\\cdot", "\\times", "×"):
        t = t.replace(token, "*")
    t = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*(\d+(?:[.,]\d+)?)", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi")
    t = t.replace("^", "**")
    t = t.replace("%", "")
    if re.fullmatch(r"-?\d+,\d+", t.strip()):
        t = t.replace(",", ".")
    else:
        t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"(\d)(sqrt|pi|\()", r"\1*\2", t)
    t = re.sub(r"(\))(sqrt|pi|\d)", r"\1*\2", t)
    t = re.sub(r"(pi)(sqrt|pi|\d|\()", r"\1*\2", t)
    if not t:
        return None
    leftover = re.sub(r"sqrt|pi|\d|\.|\+|\-|\*|/|\(|\)|e|E", "", t)
    if leftover:
        return None
    if "**" in t:
        # Avoid teaching exponent-heavy programs to a small GPT-2 unless they
        # were already simple enough to survive parse_number.
        return None
    return t


def extract_equation_steps_for_pot(solution_text: str, gold_num: Optional[float]) -> list[tuple[str, float]]:
    """Extract verified arithmetic expressions from the gold solution."""
    body = remove_final_answer_tail(solution_text)
    candidates: list[tuple[int, str, float]] = []
    pattern = re.compile(
        r"([\-]?\d[\d\s.,()+\-*/×%\\frac{}sqrt^]*?)\s*=\s*"
        r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?)"
    )
    for idx, m in enumerate(pattern.finditer(body)):
        raw_expr = m.group(1).strip()
        raw_rhs = m.group(2).strip()
        expr_py = expression_to_python(raw_expr)
        if not expr_py:
            continue
        lhs_val = parse_number(raw_expr)
        rhs_val = parse_number(raw_rhs)
        if lhs_val is None or rhs_val is None:
            continue
        if abs(lhs_val - rhs_val) > 1e-6 * max(1.0, abs(rhs_val)):
            continue
        candidates.append((idx, expr_py, rhs_val))

    if gold_num is not None:
        candidates.sort(key=lambda item: (abs(item[2] - gold_num), -item[0]))
    chosen = sorted(candidates[:FDD_MAX_EQUATION_STEPS], key=lambda item: item[0])

    out: list[tuple[str, float]] = []
    seen = set()
    for _, expr_py, val in chosen:
        if expr_py in seen:
            continue
        seen.add(expr_py)
        out.append((expr_py, val))
    return out


def has_arithmetic_signal(text: str) -> bool:
    """True when a line contains a real computation signal, not only a number."""
    return bool(
        re.search(r"(\+|\-|\*|/|×|%|\\frac|\\sqrt|\^)", text or "")
        or re.search(r"\d\s*[xX](?![A-Za-zÀ-ỹ])|(?<![A-Za-zÀ-ỹ])[xX]\s*/|(?<![A-Za-zÀ-ỹ])[xX]\s*\*", text or "")
    )


def contains_variable_x(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-zÀ-ỹ])[xX](?![A-Za-zÀ-ỹ])", text or ""))


def strip_math_delimiters(text: str) -> str:
    out = normalize_space(text)
    out = out.replace("$", "")
    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("\\cdot", "×").replace("\\times", "×")
    out = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", out)
    return normalize_space(out).strip(" .。;；")


def target_line_is_clean(line: str, require_variable: bool = False) -> bool:
    line = strip_math_delimiters(line)
    if not line or len(line) > 180:
        return False
    if any(re.search(pat, line, flags=re.IGNORECASE) for pat in ANSWER_MARKER_PATTERNS):
        return False
    if re.search(r"<[^>]+>|```|print\s*\(", line, flags=re.IGNORECASE):
        return False
    if require_variable and not contains_variable_x(line):
        return False
    if re.search(r"\d", line) and not has_arithmetic_signal(line):
        return False
    return True


def is_direct_x_answer_assignment(line: str, gold_answer: str) -> bool:
    line = strip_math_delimiters(line)
    m = re.fullmatch(r"[xX]\s*=\s*(.+)", line)
    if not m:
        return False
    rhs_num = parse_number(m.group(1))
    gold_num = parse_number(gold_answer)
    re_val = rel_error(rhs_num, gold_num)
    return re_val is not None and re_val <= 0.01


def split_reasoning_clauses(text: str) -> list[str]:
    body = remove_final_answer_tail(text)
    pieces = re.split(
        r"(?<=[.!?。])\s+|\n+|;|；|(?<=\])\s+|(?<=\))\s+",
        body,
    )
    out: list[str] = []
    for piece in pieces:
        for clause in re.split(r",|，|\s+\bnên\b\s+|\s+\bdo đó\b\s+|\s+\bvậy\b\s+|\s+\bkhi đó\b\s+", piece, flags=re.IGNORECASE):
            clause = strip_math_delimiters(clause)
            if clause:
                out.append(clause)
    return out


def extract_clean_equation_lines(
    response_vi: str,
    max_lines: int = 3,
    require_variable: bool = False,
    max_chars: int = 300,
) -> list[str]:
    lines: list[str] = []
    seen = set()
    for clause in split_reasoning_clauses(response_vi):
        if not re.search(r"\d|\\frac|\\sqrt|=", clause):
            continue
        if require_variable and not contains_variable_x(clause):
            continue
        if not target_line_is_clean(clause, require_variable=require_variable):
            continue
        if re.search(r"\d", clause) and not has_arithmetic_signal(clause):
            continue

        # Keep the local computation rather than the full Vietnamese sentence.
        if len(clause) > 120:
            op = re.search(r"=|\+|\-|\*|/|×|%|\\frac|\\sqrt|\^", clause)
            if op:
                left = max(0, op.start() - 55)
                right = min(len(clause), op.end() + 80)
                clause = clause[left:right].strip(" .。,:;；")
        key = clause.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(clause)
        if len(lines) >= max_lines:
            break

    total = 0
    kept: list[str] = []
    for line in lines:
        add_len = len(line) + (1 if kept else 0)
        if total + add_len > max_chars:
            break
        kept.append(line)
        total += add_len
    return kept


def extract_variable_equation_lines(response_vi: str, gold_answer: str) -> Optional[tuple[str, str]]:
    candidates = extract_clean_equation_lines(
        response_vi,
        max_lines=8,
        require_variable=True,
        max_chars=700,
    )
    if not candidates:
        return None

    equation = None
    solve_step = None
    for line in candidates:
        if is_direct_x_answer_assignment(line, gold_answer):
            if solve_step is None:
                solve_step = line
            continue
        if equation is None and "=" in line and has_arithmetic_signal(line):
            equation = line
            continue
        if equation is not None and solve_step is None and "=" in line:
            solve_step = line
            break

    if equation is None:
        return None
    if solve_step is None:
        solve_step = equation
    return equation, solve_step


def build_fdd_pot_target(original: str, gold_answer: str) -> Optional[str]:
    gold_num = parse_number(gold_answer)
    if gold_num is None:
        return None

    steps = extract_equation_steps_for_pot(original, gold_num)
    lines = [
        "# Chuong trinh Python ngan gon de tinh dap an",
    ]
    last_step_name = None
    for i, (expr_py, value) in enumerate(steps, 1):
        step_name = f"step_{i}"
        lines.append(f"{step_name} = {expr_py}")
        last_step_name = step_name if abs(value - gold_num) <= 1e-6 * max(1.0, abs(gold_num)) else last_step_name

    if last_step_name:
        lines.append(f"answer = {last_step_name}")
    else:
        return build_equation_short_target(original) or build_answer_only_target(original)

    if len(lines) > FDD_PROGRAM_MAX_LINES:
        lines = lines[: FDD_PROGRAM_MAX_LINES - 1]

    program = "\n".join(lines)
    return normalize_space(f"{program}\n{ANSWER_ANCHOR} {gold_answer}")


def format_answer_line(gold_answer: str) -> str:
    return normalize_space(f"{ANSWER_ANCHOR} {cleanup_answer_string(gold_answer)}")


def build_answer_only_target(response_vi: str) -> Optional[str]:
    """
    Return exactly:
    ####đáp án là: <answer>
    """
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    target = format_answer_line(gold_answer)
    assert count_answer_anchors(target) == 1
    return target


def build_equation_short_target(response_vi: str) -> Optional[str]:
    """
    Extract up to 2-3 useful arithmetic/equation lines from the gold response.
    End with exactly one:
    ####đáp án là: <answer>
    """
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None

    lines = extract_clean_equation_lines(response_vi, max_lines=3, require_variable=False, max_chars=300)
    if not lines:
        return None

    body = "\n".join(lines[:3]).strip()
    if not body:
        return None

    target = normalize_space(f"Lời giải ngắn:\n{body}\n{format_answer_line(gold_answer)}")
    if len(target) > 380:
        clipped = "\n".join(lines[:2]).strip()
        target = normalize_space(f"Lời giải ngắn:\n{clipped}\n{format_answer_line(gold_answer)}")
    assert count_answer_anchors(target) == 1
    return target


def build_symbolic_equation_short_target(response_vi: str) -> Optional[str]:
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    lines = extract_clean_equation_lines(response_vi, max_lines=2, require_variable=False, max_chars=260)
    if not lines:
        return None
    target = normalize_space(f"Lời giải ngắn:\n" + "\n".join(lines) + f"\n{format_answer_line(gold_answer)}")
    if len(target) > 340:
        return None
    assert count_answer_anchors(target) == 1
    return target


def build_variable_equation_target(response_vi: str) -> Optional[str]:
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    found = extract_variable_equation_lines(response_vi, gold_answer)
    if found is None:
        return None
    equation, solve_step = found
    if is_direct_x_answer_assignment(equation, gold_answer):
        return None
    target = normalize_space(
        "Phương trình: "
        f"{equation}\n"
        "Giải: "
        f"{solve_step}\n"
        f"x = {cleanup_answer_string(gold_answer)}\n"
        f"{format_answer_line(gold_answer)}"
    )
    if len(target) > 420:
        return None
    assert count_answer_anchors(target) == 1
    return target


def build_fdd_lite_pot_target(response_vi: str) -> Optional[str]:
    """
    Build a short Program-of-Thought-style target from verified gold equations.
    This is used as a training target only. Inference does not execute generated
    Python by default.
    """
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    gold_num = parse_number(gold_answer)
    if gold_num is None:
        return None

    steps = extract_equation_steps_for_pot(response_vi, gold_num)[:FDD_MAX_EQUATION_STEPS]
    if not steps:
        return build_answer_only_target(response_vi)

    lines: list[str] = []
    final_step_name = None
    for i, (expr_py, value) in enumerate(steps, 1):
        step_name = f"s{i}"
        lines.append(f"{step_name} = {expr_py}")
        if abs(value - gold_num) <= 1e-6 * max(1.0, abs(gold_num)):
            final_step_name = step_name

    # Do not directly set answer = gold when useful extracted steps exist but
    # none of them reaches the gold value. The final answer line still provides
    # the supervised answer anchor.
    if final_step_name:
        lines.append(f"answer = {final_step_name}")

    target = normalize_space("\n".join(lines + [format_answer_line(gold_answer)]))
    if len(target) > 380:
        keep = lines[:2]
        if final_step_name and f"answer = {final_step_name}" not in keep:
            keep.append(f"answer = {final_step_name}")
        target = normalize_space("\n".join(keep + [format_answer_line(gold_answer)]))
    assert count_answer_anchors(target) == 1
    return target


def build_fdd_pot_compact_target(response_vi: str) -> Optional[str]:
    """
    Compact FDD/PoT target: prefer verified arithmetic program steps, then
    equation snippets, and use answer-only only when no reasoning trace exists.
    """
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    gold_num = parse_number(gold_answer)
    if gold_num is None:
        return None

    steps = extract_equation_steps_for_pot(response_vi, gold_num)[:FDD_MAX_EQUATION_STEPS]
    if steps:
        lines: list[str] = []
        last_step_name = None
        for i, (expr_py, value) in enumerate(steps, 1):
            step_name = f"step_{i}"
            lines.append(f"{step_name} = {expr_py}")
            if abs(value - gold_num) <= 1e-6 * max(1.0, abs(gold_num)):
                last_step_name = step_name

        if last_step_name:
            lines.append(f"answer = {last_step_name}")
        else:
            lines = []

        if lines:
            target = normalize_space("\n".join(lines + [format_answer_line(gold_answer)]))
            if len(target) <= 520 and count_answer_anchors(target) == 1:
                return target

    equation_target = build_equation_short_target(response_vi)
    if equation_target:
        return equation_target
    return build_answer_only_target(response_vi)


def build_original_fdd_pot_restored_target(response_vi: str) -> Optional[str]:
    """Restore the original FDD/PoT-style supervision without print(answer)."""
    gold_answer = extract_answer(response_vi)
    if gold_answer is None:
        return None
    gold_num = parse_number(gold_answer)
    if gold_num is not None:
        steps = extract_equation_steps_for_pot(response_vi, gold_num)[:FDD_MAX_EQUATION_STEPS]
        if steps:
            lines = ["# Chuong trinh tinh ngan gon"]
            last_step_name = None
            for i, (expr_py, value) in enumerate(steps, 1):
                step_name = f"step_{i}"
                lines.append(f"{step_name} = {expr_py}")
                if abs(value - gold_num) <= 1e-6 * max(1.0, abs(gold_num)):
                    last_step_name = step_name
            if last_step_name:
                lines.append(f"answer = {last_step_name}")
                target = normalize_space("\n".join(lines + [format_answer_line(gold_answer)]))
                if count_answer_anchors(target) == 1:
                    return target

    body = equation_focused_reasoning(response_vi, max_chars=700, max_parts=6) or shorten_reasoning(response_vi)
    if body:
        target = normalize_space(f"{body}\n{format_answer_line(gold_answer)}")
        if count_answer_anchors(target) == 1:
            return target
    return build_answer_only_target(response_vi)


def build_target_for_record(rec: dict) -> Optional[str]:
    original = normalize_space(str(rec.get("response_vi", rec.get("response_vi_original", ""))).strip())
    if not original:
        return None

    problem_type = str(rec.get("type") or "")
    target = None
    if TYPE_ROUTING and ("SV" in problem_type or "FOBAR" in problem_type):
        target = build_fdd_lite_pot_target(original)
        if target:
            return target

    target = build_equation_short_target(original)
    if target:
        return target
    return build_answer_only_target(original)


SV_FOBAR_TYPES = {"GSM_SV", "GSM_FOBAR", "MATH_SV", "MATH_FOBAR"}
GSM_EQUATION_TYPES = {"GSM_REPHRASED", "GSM_ANSAUG"}
MATH_EQUATION_TYPES = {"MATH_REPHRASED", "MATH_ANSAUG"}


def build_type_specific_equation_target(rec: dict) -> tuple[Optional[str], str]:
    original = normalize_space(str(rec.get("response_vi", rec.get("response_vi_original", ""))).strip())
    if not original:
        return None, "missing_response"

    problem_type = str(rec.get("type") or "").upper()

    if problem_type in SV_FOBAR_TYPES:
        target = build_variable_equation_target(original)
        if target:
            return target, "sv_fobar_variable_equation"
        target = build_equation_short_target(original)
        if target:
            return target, "sv_fobar_equation_short_fallback"
        target = build_answer_only_target(original)
        return target, "fallback_answer_only"

    if problem_type in GSM_EQUATION_TYPES or problem_type.startswith("GSM_"):
        target = build_equation_short_target(original)
        if target:
            return target, "gsm_equation_short"
        target = build_answer_only_target(original)
        return target, "fallback_answer_only"

    if problem_type in MATH_EQUATION_TYPES or problem_type.startswith("MATH_"):
        target = build_symbolic_equation_short_target(original)
        if target:
            return target, "math_symbolic_equation_short"
        target = build_answer_only_target(original)
        return target, "math_answer_only" if target else "fallback_answer_only"

    target = build_equation_short_target(original)
    if target:
        return target, "other_equation_short"
    target = build_answer_only_target(original)
    return target, "fallback_answer_only"


def build_target(
    rec: dict,
    train_style: str = TRAIN_STYLE,
    filter_no_answer: bool = FILTER_NO_ANSWER,
    filter_non_numeric_gold: bool = FILTER_NON_NUMERIC_GOLD,
) -> Optional[str]:
    if train_style == "preprocessed_target":
        target = normalize_space(str(rec.get(PREPROCESSED_TARGET_FIELD, "") or ""))
        if not target:
            target = normalize_space(str(rec.get("target_answer_only", "") or ""))
        if not target:
            return None
        return target

    if train_style in {
        "fdd_lite_type_routing",
        "fdd_pot_compact",
        "type_specific_equation_targets",
        "original_fdd_pot_restored",
    } and str(rec.get("_target", "") or "").strip():
        return normalize_space(str(rec["_target"]))

    original = normalize_space(str(rec.get("response_vi", rec.get("response_vi_original", ""))).strip())
    gold_answer = extract_answer(original)

    if gold_answer is None:
        if filter_no_answer:
            return None
        return original

    if filter_non_numeric_gold and parse_number(gold_answer) is None:
        return None

    final_line = format_answer_line(gold_answer)

    if train_style == "original":
        return original
    if train_style == "standardized":
        body = remove_final_answer_tail(original)
        return normalize_space(f"{body}\n{final_line}" if body else final_line)
    if train_style == "short_solution":
        body = shorten_reasoning(original)
        return normalize_space(f"{body}\n{final_line}" if body else final_line)
    if train_style in {"answer_stub", "answer_only", "answer_focused"}:
        return final_line
    if train_style == "equation_focused":
        body = equation_focused_reasoning(original)
        return normalize_space(f"{body}\n{final_line}" if body else final_line)
    if train_style == "fdd_pot":
        return build_fdd_pot_target(original, gold_answer)
    if train_style == "fdd_pot_compact":
        return build_fdd_pot_compact_target(original)
    if train_style == "original_fdd_pot_restored":
        return build_original_fdd_pot_restored_target(original)
    if train_style == "fdd_lite_type_routing":
        return build_target_for_record(rec)
    if train_style == "type_specific_equation_targets":
        target, _ = build_type_specific_equation_target(rec)
        return target
    if train_style == "tagged_short_solution":
        body = equation_focused_reasoning(original, max_chars=450, max_parts=4) or shorten_reasoning(original)
        if not body:
            body = "Ta tính theo dữ kiện trong đề bài."
        return normalize_space(
            "<think>\n"
            f"{body}\n"
            "</think>\n"
            "<answer>\n"
            f"{gold_answer}\n"
            "</answer>"
        )
    if train_style == "tagged_equation_focused":
        body = equation_focused_reasoning(
            original,
            max_chars=TAGGED_REASONING_MAX_CHARS,
            max_parts=TAGGED_REASONING_MAX_PARTS,
        ) or shorten_reasoning(original, max_chars=TAGGED_REASONING_MAX_CHARS)
        if not body:
            body = "Ta tính theo dữ kiện trong đề bài."
        return normalize_space(
            "<think>\n"
            f"{body}\n"
            "</think>\n"
            "<answer>\n"
            f"{gold_answer}\n"
            "</answer>"
        )
    if train_style == "tagged_equation_minimal":
        body = minimal_equation_reasoning(
            original,
            max_chars=TAGGED_REASONING_MAX_CHARS,
            max_parts=TAGGED_REASONING_MAX_PARTS,
        )
        if not body:
            body = "Ta tính theo dữ kiện trong đề bài."
        return normalize_space(
            "<think>\n"
            f"{body}\n"
            "</think>\n"
            "<answer>\n"
            f"{gold_answer}\n"
            "</answer>"
        )
    if train_style == "tagged_answer_only":
        return normalize_space(
            "<think>\n"
            "Ta tính theo dữ kiện trong đề bài.\n"
            "</think>\n"
            "<answer>\n"
            f"{gold_answer}\n"
            "</answer>"
        )

    raise ValueError(f"Unknown TRAIN_STYLE: {train_style}")


def final_answer_tail(target_text: str) -> Optional[str]:
    """Return the final answer line, including a leading newline if needed."""
    lines = [line.strip() for line in normalize_space(target_text).split("\n") if line.strip()]
    for line in reversed(lines):
        if extract_answer(line) is not None:
            marker_starts = []
            for pat in ANSWER_MARKER_PATTERNS:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    marker_starts.append(m.start())
            if marker_starts:
                return line[min(marker_starts):].strip()
            return line
    return None

# %% [markdown]
# ## 3. Light EDA

# %%
def percentiles(values: list[int | float], qs: Iterable[int] = (0, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float]:
    if not values:
        return {str(q): 0.0 for q in qs}
    xs = sorted(values)
    out = {}
    for q in qs:
        pos = (len(xs) - 1) * q / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            val = xs[lo]
        else:
            val = xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
        out[str(q)] = float(val)
    return out


def missing_count(records: list[dict], key: str) -> int:
    return sum(1 for r in records if not str(r.get(key, "")).strip())


def approximate_token_length(rec: dict, tok: Any) -> int:
    prompt = build_prompt(rec)
    target = build_target(rec, filter_no_answer=False, filter_non_numeric_gold=False)
    if target is None:
        target = normalize_space(str(rec.get("response_vi") or rec.get("response_vi_original") or ""))
    ids = tok(prompt + "\n" + target, add_special_tokens=False)["input_ids"]
    return len(ids) + 1  # EOS


def eda_for_split(records: list[dict], split: str, tok: Any, max_length: int) -> dict:
    q_texts = [get_query_text(r) for r in records]
    responses = [
        normalize_space(str(
            r.get("response_vi")
            or r.get("response_vi_original")
            or r.get(PREPROCESSED_TARGET_FIELD)
            or r.get("target_answer_only")
            or ""
        ))
        for r in records
    ]
    q_lens = [len(x) for x in q_texts]
    r_lens = [len(x) for x in responses if x]

    token_lens = []
    over_max = 0
    extractable = 0
    parseable = 0
    for rec in tqdm(records, desc=f"EDA {split}", leave=False):
        gold_source = (
            rec.get("response_vi")
            or rec.get("response_vi_original")
            or rec.get(PREPROCESSED_TARGET_FIELD)
            or rec.get("target_answer_only")
        )
        if extract_answer(gold_source) is not None:
            extractable += 1
        ans = extract_answer(gold_source)
        if parse_number(ans) is not None:
            parseable += 1

        n_tokens = approximate_token_length(rec, tok)
        token_lens.append(n_tokens)
        over_max += int(n_tokens > max_length)

    duplicated_questions = len(q_texts) - len(set(q_texts))

    return {
        "split": split,
        "n": len(records),
        "type_counts": dict(Counter(str(r.get("type", "missing")) for r in records).most_common()),
        "missing_query_vi": missing_count(records, "query_vi"),
        "missing_query_text": sum(1 for r in records if not get_query_text(r)),
        "missing_response_vi": missing_count(records, "response_vi"),
        "missing_sft_target": sum(1 for r in records if build_target(r, filter_no_answer=False, filter_non_numeric_gold=False) is None),
        "missing_type": missing_count(records, "type"),
        "duplicate_questions": duplicated_questions,
        "question_char_length_percentiles": percentiles(q_lens),
        "response_char_length_percentiles": percentiles(r_lens),
        "approx_token_length_percentiles": percentiles(token_lens),
        "gold_extractable_answers": extractable,
        "gold_parseable_numeric_answers": parseable,
        "prompt_target_over_max_length": over_max,
        "prompt_target_over_max_length_pct": over_max / len(records) if records else 0.0,
    }


# parse_number is defined in the next cell, then EDA is executed.

# %% [markdown]
# ## 4. Answer Extraction and Evaluation

# %%
def parse_number(s: Optional[str]) -> Optional[float]:
    """Best-effort parser for finite scalar numeric answers."""
    if s is None:
        return None

    original = str(s).strip()
    if not original:
        return None

    t = original
    t = t.replace("\u2212", "-")
    t = t.replace("−", "-")
    t = t.strip().strip("\"'“”‘’`")

    # Remove obvious wrappers and punctuation that often surround final answers.
    t = t.strip("$ ").strip("\"'“”‘’`")
    t = t.rstrip(".。;；")

    # Strip variable assignment: x = 5 -> 5
    m = re.match(r"^[A-Za-z_]\w*\s*=\s*(.+)$", t)
    if m:
        t = m.group(1).strip()

    # Reject tuples / intervals / lists / multiple-answer expressions early.
    compact0 = re.sub(r"\s+", "", t)
    if (
        (compact0.startswith("(") and compact0.endswith(")") and re.search(r"\d\s*,\s*\d", t))
        or (compact0.startswith("[") and compact0.endswith("]"))
        or re.search(r"\b(?:hoặc|và|and|or)\b", t, flags=re.IGNORECASE)
    ):
        return None

    # LaTeX cleanup.
    compact_latex = re.sub(r"\s+", "", t.replace("$", ""))
    mixed = re.fullmatch(
        r"([+-]?\d+)\\(?:d|t)?frac\{([+-]?\d+(?:[.,]\d+)?)\}\{([+-]?\d+(?:[.,]\d+)?)\}",
        compact_latex,
    )
    if mixed:
        whole = float(mixed.group(1).replace(",", "."))
        numerator = float(mixed.group(2).replace(",", "."))
        denominator = float(mixed.group(3).replace(",", "."))
        if denominator == 0:
            return None
        sign = -1.0 if whole < 0 else 1.0
        val = sign * (abs(whole) + numerator / denominator)
        return val if math.isfinite(val) else None

    for _ in range(3):
        new = re.sub(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r"(\1)", t)
        if new == t:
            break
        t = new

    t = re.sub(r"\\text\{[^}]*\}", "", t)
    t = re.sub(r"\\mathrm\{[^}]*\}", "", t)
    t = t.replace("$", "")

    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right"):
        t = t.replace(token, "")
    for token in ("\\cdot", "\\times"):
        t = t.replace(token, "*")

    t = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*(\d+(?:[.,]\d+)?)", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi")

    # Percent signs are treated as units, not divided by 100, because many gold
    # answers in word-problem datasets expect "50" rather than "0.5".
    t = t.replace("%", "")

    # If the answer is a single leading number/fraction followed by units, keep
    # the number only. Do not accept tails that contain another digit.
    unit_match = re.match(
        r"^\s*(-?\d+(?:[.,]\d+)?(?:\s*/\s*-?\d+(?:[.,]\d+)?)?)\s+([^\d]+)$",
        t,
    )
    if unit_match and not re.search(r"\d", unit_match.group(2)):
        t = unit_match.group(1)

    # Vietnamese decimal: 1,5 -> 1.5. English thousands: 1,000 -> 1000.
    has_period = "." in t
    n_commas = t.count(",")
    if n_commas == 1 and not has_period and re.search(r"\d,\d", t):
        if re.fullmatch(r"\s*-?\d+,\d+\s*", t):
            t = re.sub(r"(?<=\d),(?=\d)", ".", t)
        else:
            t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    elif n_commas >= 1:
        t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)

    # Implicit multiplication after decimal cleanup.
    t = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", t)
    t = re.sub(r"(\))\s*(sqrt|pi|\d)", r"\1*\2", t)
    t = re.sub(r"(pi)\s*(sqrt|pi|\d|\()", r"\1*\2", t)
    t = re.sub(r"\s+", "", t)

    if not t or "," in t:
        return None

    leftover = re.sub(r"sqrt|pi|\d|\.|\+|\-|\*|/|\(|\)|\^|e|E", "", t)
    if leftover:
        return None

    t = t.replace("^", "**")

    try:
        val = eval(t, {"__builtins__": {}}, SAFE_NS)
    except Exception:
        return None

    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        val = float(val)
        return val if math.isfinite(val) else None
    return None


def rel_error(pred: Optional[float], gold: Optional[float]) -> Optional[float]:
    if pred is None or gold is None:
        return None
    return abs(pred - gold) / max(1.0, abs(gold))


def score_one(re_val: Optional[float], extractable: bool) -> int:
    if not extractable or re_val is None:
        return 0
    if re_val <= 0.01:
        return 10
    if re_val <= 0.10:
        return 5
    if re_val <= 0.50:
        return 1
    return 0


TAG_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)
TAG_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def extract_tagged_answer(text: str) -> Optional[str]:
    matches = TAG_ANSWER_RE.findall(text or "")
    if len(matches) != 1:
        return None
    ans = strip_trailing_special_artifacts(matches[0].strip())
    return ans or None


def has_repeated_grpo_tags(text: str) -> bool:
    lower = (text or "").lower()
    return any(lower.count(tag) > 1 for tag in ("<think>", "</think>", "<answer>", "</answer>"))


def calculate_format_reward(text: str) -> float:
    text = text or ""
    think_matches = list(TAG_THINK_RE.finditer(text))
    answer_matches = list(TAG_ANSWER_RE.finditer(text))
    lower = text.lower()
    think_tag_count = lower.count("<think>") + lower.count("</think>")
    answer_tag_count = lower.count("<answer>") + lower.count("</answer>")

    if has_repeated_grpo_tags(text):
        return -0.5
    if answer_tag_count and len(answer_matches) != 1:
        return -0.5
    if think_tag_count and len(think_matches) != 1:
        return -0.5
    if len(answer_matches) != 1:
        return 0.0

    answer_text = strip_trailing_special_artifacts(answer_matches[0].group(1).strip())
    if not answer_text:
        return 0.0

    answer_is_numeric = parse_number(answer_text) is not None
    if GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER and not answer_is_numeric:
        return 0.0

    if len(think_matches) == 1:
        if think_matches[0].start() < think_matches[0].end() <= answer_matches[0].start():
            return 1.0
        return -0.5

    if len(think_matches) == 0:
        # Partial credit only for a usable numeric answer block. This keeps
        # malformed or empty answer tags from dominating sparse correctness.
        return 0.5
    return -0.5


def calculate_correctness_reward(completion_text: str, gold_answer: str) -> float:
    pred_answer = extract_tagged_answer(completion_text)
    if pred_answer is None:
        return 0.0
    pred_num = parse_number(pred_answer)
    gold_num = parse_number(gold_answer)
    re_val = rel_error(pred_num, gold_num)
    if re_val is None:
        return 0.0
    if not GRPO_USE_PARTIAL_NUMERIC_REWARD:
        return 1.0 if re_val <= 0.01 else 0.0
    if re_val <= 0.01:
        return 1.0
    if re_val <= 0.10:
        return 0.5
    if re_val <= 0.50:
        return 0.1
    return 0.0


def compute_grpo_reward(completion_text: str, gold_answer: str) -> dict:
    correctness_reward = calculate_correctness_reward(completion_text, gold_answer)
    format_reward = calculate_format_reward(completion_text)
    pred_answer = extract_tagged_answer(completion_text)
    total_reward = (
        GRPO_CORRECTNESS_WEIGHT * correctness_reward
        + GRPO_FORMAT_WEIGHT * format_reward
    )
    return {
        "total": float(total_reward),
        "correctness": float(correctness_reward),
        "format": float(format_reward),
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
    }


def convert_tagged_to_vietnamese_output(text: str) -> Optional[str]:
    think_matches = TAG_THINK_RE.findall(text or "")
    answer = extract_tagged_answer(text or "")
    if not think_matches or answer is None:
        return None
    reasoning = strip_trailing_special_artifacts(normalize_space(think_matches[0]))
    answer = strip_trailing_special_artifacts(normalize_space(answer))
    if not reasoning:
        reasoning = "Bài giải được trình bày trong phần suy luận của mô hình."
    return normalize_space(f"Lời giải: {reasoning}\nĐáp án là: {answer}")


def align_predictions_with_gold(
    pred_items: list[dict],
    gold_items: list[dict],
) -> list[tuple[dict, dict]]:
    pred_has_id = all("id" in x for x in pred_items)
    gold_has_id = all("id" in x for x in gold_items)

    if pred_has_id and gold_has_id:
        pred_map = {str(x["id"]): x for x in pred_items}
        pairs = []
        missing = []
        for g in gold_items:
            gid = str(g["id"])
            if gid not in pred_map:
                missing.append(gid)
            else:
                pairs.append((pred_map[gid], g))
        if missing:
            raise ValueError(f"Prediction thiếu {len(missing)} id, ví dụ: {missing[:5]}")
        return pairs

    if len(pred_items) != len(gold_items):
        raise ValueError(
            f"Số lượng prediction ({len(pred_items)}) khác số lượng gold ({len(gold_items)})."
        )
    return list(zip(pred_items, gold_items))


def evaluate(pred_items: list[dict], gold_items: list[dict]) -> dict:
    pairs = align_predictions_with_gold(pred_items, gold_items)

    rows = []
    total = 0
    bucket10 = bucket5 = bucket1 = bucket0 = 0
    extractable = 0
    numeric_pairs = 0
    exact_zero_error = 0
    rel_errors = []

    type_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "raw_score": 0,
            "extractable": 0,
            "numeric_pairs": 0,
            "exact_count": 0,
            "exact_zero_error_count": 0,
            "buckets": {"10": 0, "5": 0, "1": 0, "0": 0},
        }
    )

    for pred_rec, gold_rec in pairs:
        gold_ans = extract_answer(gold_rec.get("response_vi"))
        pred_ans = extract_answer(pred_rec.get("model_output"))

        is_extractable = pred_ans is not None
        extractable += int(is_extractable)

        gold_num = parse_number(gold_ans)
        pred_num = parse_number(pred_ans)
        re_val = rel_error(pred_num, gold_num)

        if gold_num is not None and pred_num is not None and re_val is not None:
            numeric_pairs += 1
            rel_errors.append(re_val)
            exact_zero_error += int(re_val <= 1e-12)

        s = score_one(re_val, is_extractable)
        total += s

        bucket10 += int(s == 10)
        bucket5 += int(s == 5)
        bucket1 += int(s == 1)
        bucket0 += int(s == 0)

        problem_type = str(gold_rec.get("type") or pred_rec.get("type") or "unknown")
        ts = type_stats[problem_type]
        ts["n"] += 1
        ts["raw_score"] += s
        ts["extractable"] += int(is_extractable)
        ts["numeric_pairs"] += int(gold_num is not None and pred_num is not None and re_val is not None)
        ts["exact_count"] += int(s == 10)
        ts["exact_zero_error_count"] += int(re_val is not None and re_val <= 1e-12)
        ts["buckets"][str(s)] += 1

        rows.append({
            "id": gold_rec.get("id", pred_rec.get("id")),
            "type": problem_type,
            "gold_answer": gold_ans,
            "pred_answer": pred_ans,
            "gold_num": gold_num,
            "pred_num": pred_num,
            "rel_error": re_val,
            "extractable": is_extractable,
            "score": s,
        })

    n = len(rows)
    max_score = n * 10
    score_by_type = {}
    for typ, stats in sorted(type_stats.items()):
        tn = stats["n"]
        stats["max_raw_score"] = tn * 10
        stats["score_10"] = stats["raw_score"] / tn if tn else 0.0
        stats["score_pct"] = stats["raw_score"] / (tn * 10) if tn else 0.0
        stats["exact_rate"] = stats["exact_count"] / tn if tn else 0.0
        stats["exact_zero_error_rate"] = stats["exact_zero_error_count"] / tn if tn else 0.0
        score_by_type[typ] = stats

    return {
        "summary": {
            "n": n,
            "raw_score": total,
            "total_score": total,
            "max_raw_score": max_score,
            "max_score": max_score,
            "score_10": total / n if n else 0.0,
            "score_pct": total / max_score if max_score else 0.0,
            "extractable": extractable,
            "numeric_pairs": numeric_pairs,
            "exact_count": bucket10,
            "exact_rate": bucket10 / n if n else 0.0,
            "exact_zero_error_count": exact_zero_error,
            "exact_zero_error_rate": exact_zero_error / n if n else 0.0,
            "buckets": {"10": bucket10, "5": bucket5, "1": bucket1, "0": bucket0},
            "rel_error_mean": sum(rel_errors) / len(rel_errors) if rel_errors else None,
            "score_by_type": score_by_type,
        },
        "rows": rows,
    }


def save_evaluation_report(
    pred_path: str | Path,
    gold_records: list[dict],
    report_path: str | Path,
) -> dict:
    pred_path = Path(pred_path)
    with pred_path.open("r", encoding="utf-8") as f:
        pred_items = json.load(f)

    result = evaluate(pred_items, gold_records)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    save_json(result, report_path)
    print(f"Wrote {report_path}")
    return result


def normalize_question_for_dedup(text: str) -> str:
    text = normalize_space(str(text).lower())
    text = re.sub(r"\s+", " ", text)
    return text


def dedup_by_question(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for rec in records:
        key = normalize_question_for_dedup(get_query_text(rec))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def filter_records_by_type_family(records: list[dict], family: str) -> list[dict]:
    if not family:
        return list(records)
    family = family.upper()
    return [
        rec for rec in records
        if str(rec.get("type") or "").upper().startswith(family)
    ]


def sample_by_type(records: list[dict], max_samples: int, weights: Optional[dict[str, float]] = None) -> list[dict]:
    if max_samples >= len(records):
        return list(records)

    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get("type") or "unknown")].append(rec)

    rng = random.Random(SEED)
    for group in groups.values():
        rng.shuffle(group)

    if weights is None:
        per_type = max(1, math.ceil(max_samples / max(1, len(groups))))
        selected = []
        for typ in sorted(groups):
            selected.extend(groups[typ][:per_type])
        if len(selected) < max_samples:
            leftovers = [rec for typ in sorted(groups) for rec in groups[typ][per_type:]]
            rng.shuffle(leftovers)
            selected.extend(leftovers[: max_samples - len(selected)])
        return selected[:max_samples]

    selected = []
    used_ids = set()
    for family, weight in weights.items():
        want = int(round(max_samples * weight))
        pool = [
            rec for typ, group in groups.items()
            if typ.upper().startswith(family.upper())
            for rec in group
        ]
        selected.extend(pool[:want])
        used_ids.update(id(rec) for rec in selected)

    leftovers = [rec for rec in records if id(rec) not in used_ids]
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max_samples - len(selected)])
    return selected[:max_samples]


def select_train_records(records: list[dict], tok: Any) -> list[dict]:
    selected = list(records)

    if TRAIN_TYPE_FILTER:
        before = len(selected)
        selected = filter_records_by_type_family(selected, TRAIN_TYPE_FILTER)
        print(f"TRAIN_TYPE_FILTER={TRAIN_TYPE_FILTER}: {before} -> {len(selected)}")

    if DEDUP_TRAIN_QUESTIONS:
        before = len(selected)
        selected = dedup_by_question(selected)
        print(f"DEDUP_TRAIN_QUESTIONS=True: {before} -> {len(selected)}")

    if SAMPLING_STYLE == "easy_first":
        selected = sorted(selected, key=lambda rec: approximate_token_length(rec, tok))
        if MAX_TRAIN_SAMPLES:
            selected = selected[:MAX_TRAIN_SAMPLES]
    elif SAMPLING_STYLE == "balanced_type" and MAX_TRAIN_SAMPLES:
        selected = sample_by_type(selected, MAX_TRAIN_SAMPLES)
    elif SAMPLING_STYLE == "gsm_heavy" and MAX_TRAIN_SAMPLES:
        selected = sample_by_type(selected, MAX_TRAIN_SAMPLES, {"GSM": 0.75})
    elif SAMPLING_STYLE == "math_heavy" and MAX_TRAIN_SAMPLES:
        selected = sample_by_type(selected, MAX_TRAIN_SAMPLES, {"MATH": 0.75})
    elif SAMPLING_STYLE == "natural":
        if MAX_TRAIN_SAMPLES:
            selected = selected[:MAX_TRAIN_SAMPLES]
    else:
        if SAMPLING_STYLE not in {"balanced_type", "gsm_heavy", "math_heavy"}:
            raise ValueError(f"Unknown SAMPLING_STYLE: {SAMPLING_STYLE}")
        print(f"SAMPLING_STYLE={SAMPLING_STYLE} with MAX_TRAIN_SAMPLES=None keeps all selected records.")

    print(
        "Selected train records:",
        len(selected),
        "| sampling_style:",
        SAMPLING_STYLE,
        "| dedup:",
        DEDUP_TRAIN_QUESTIONS,
    )
    print("Selected train type counts:", dict(Counter(str(r.get("type", "missing")) for r in selected).most_common()))
    return selected


train_records = select_train_records(train_records_all, tokenizer)


def attach_prebuilt_targets(records: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Attach `_target` for the final FDD-lite candidate and report target mix."""
    if INFERENCE_VOTING and not RUN_TRAIN:
        return records, {"enabled": False, "reason": "inference_voting_frozen_checkpoint"}

    if TRAIN_STYLE not in {
        "fdd_lite_type_routing",
        "fdd_pot_compact",
        "type_specific_equation_targets",
        "original_fdd_pot_restored",
    }:
        return records, {"enabled": False}

    prepared: list[dict] = []
    skipped = 0
    target_kinds: Counter[str] = Counter()
    anchor_errors = 0
    char_lengths: list[int] = []

    for rec in records:
        original = normalize_space(str(rec.get("response_vi", rec.get("response_vi_original", ""))).strip())
        gold_answer = extract_answer(original)
        if gold_answer is None:
            skipped += 1
            continue
        if FILTER_NON_NUMERIC_GOLD and parse_number(gold_answer) is None:
            skipped += 1
            continue
        target_kind = ""
        if TRAIN_STYLE == "type_specific_equation_targets":
            target, target_kind = build_type_specific_equation_target(rec)
        elif TRAIN_STYLE == "original_fdd_pot_restored":
            target = build_original_fdd_pot_restored_target(original)
            target_kind = "original_fdd_pot_restored"
        elif TRAIN_STYLE == "fdd_pot_compact":
            target = build_fdd_pot_compact_target(original)
            target_kind = "fdd_pot_compact"
        else:
            target = build_target_for_record(rec)
            target_kind = "fdd_lite_type_routing"
        if not target:
            skipped += 1
            continue
        if count_answer_anchors(target) != 1:
            anchor_errors += 1
            skipped += 1
            continue

        new_rec = dict(rec)
        new_rec["_target"] = target
        new_rec["_target_kind"] = target_kind
        if target.startswith(ANSWER_ANCHOR):
            target_kinds[target_kind or "answer_only"] += 1
        elif re.search(r"^(?:s|step_)\d+\s*=", target, flags=re.MULTILINE):
            target_kinds["fdd_pot_compact" if TRAIN_STYLE == "fdd_pot_compact" else "fdd_lite_pot"] += 1
        else:
            target_kinds[target_kind or "equation_short"] += 1
        char_lengths.append(len(target))
        prepared.append(new_rec)

    sv_fobar_variable = target_kinds.get("sv_fobar_variable_equation", 0)
    gsm_equation_short = target_kinds.get("gsm_equation_short", 0)
    math_answer_only = target_kinds.get("math_answer_only", 0)
    fallback_answer_only = target_kinds.get("fallback_answer_only", 0)

    report = {
        "enabled": True,
        "input_records": len(records),
        "kept_records": len(prepared),
        "skipped_records": skipped,
        "anchor_errors": anchor_errors,
        "target_kinds": dict(target_kinds),
        "sv_fobar_variable_equation_targets": sv_fobar_variable,
        "gsm_equation_short_targets": gsm_equation_short,
        "math_answer_only_targets": math_answer_only,
        "fallback_answer_only_targets": fallback_answer_only,
        "target_char_percentiles": percentiles(char_lengths),
    }
    print("Target preparation report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return prepared, report


train_records, target_build_report = attach_prebuilt_targets(train_records)
valid_records = valid_records_all[:MAX_VALID_SAMPLES] if MAX_VALID_SAMPLES else valid_records_all

print("train:", len(train_records), "/", len(train_records_all))
print("valid:", len(valid_records), "/", len(valid_records_all))


# Execute EDA after parse_number is available.
eda_report = {
    "train": eda_for_split(train_records, "train", tokenizer, MAX_LENGTH),
    "valid": eda_for_split(valid_records, "valid", tokenizer, MAX_LENGTH),
}
print("EDA summary:")
print(json.dumps(eda_report, ensure_ascii=False, indent=2)[:6000])
update_experiment_report(eda=eda_report)
update_experiment_report(target_build_report=target_build_report)
print("Wrote", EXPERIMENT_REPORT_PATH)

# %% [markdown]
# ## 5. Dataset and Collator

# %%
def clamp_token_ids(ids: list[int], vocab_size: int) -> list[int]:
    return [tok if 0 <= tok < vocab_size else SAFE_EOS_ID for tok in ids]


def ensure_eos(ids: list[int], eos_id: int, max_len: int) -> list[int]:
    if not ids:
        return [eos_id]
    if ids[-1] == eos_id:
        return ids
    if len(ids) < max_len:
        return ids + [eos_id]
    return ids[:-1] + [eos_id]


def find_last_subsequence(haystack: list[int], needle: list[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def make_target_labels(
    target_ids: list[int],
    target_text: str,
    tok: Any,
    vocab_size: int,
    loss_style: str,
) -> list[int]:
    if loss_style in {"full_target", "answer_weighted", "light_answer_weighted", "full_target_or_light_weighted"}:
        return list(target_ids)

    if loss_style != "answer_line_only":
        raise ValueError(f"Unknown LOSS_STYLE: {loss_style}")

    labels = [-100] * len(target_ids)
    answer_line = final_answer_tail(target_text)
    if not answer_line:
        if target_ids and target_ids[-1] == SAFE_EOS_ID:
            labels[-1] = SAFE_EOS_ID
        return labels

    supervise_ids = tok("\n" + answer_line, add_special_tokens=False)["input_ids"]
    supervise_ids = ensure_eos(clamp_token_ids(supervise_ids, vocab_size), SAFE_EOS_ID, len(target_ids))
    start = find_last_subsequence(target_ids, supervise_ids)

    if start is None and supervise_ids and supervise_ids[0] != SAFE_EOS_ID:
        start = find_last_subsequence(target_ids, supervise_ids[1:])
        if start is not None:
            supervise_ids = supervise_ids[1:]

    if start is not None:
        for offset, token_id in enumerate(supervise_ids):
            labels[start + offset] = token_id
    elif target_ids and target_ids[-1] == SAFE_EOS_ID:
        # Fallback: supervise the last answer-like suffix when exact token
        # matching fails after truncation.
        n = min(len(supervise_ids), len(target_ids))
        labels[-n:] = target_ids[-n:]

    return labels


def build_target_loss_weights(target_ids: list[int], target_text: str, tok: Any) -> list[float]:
    weights = [float(BASE_RESPONSE_WEIGHT)] * len(target_ids)

    # Emphasize local computation tokens, but let the final answer span win.
    for idx, token_id in enumerate(target_ids):
        if token_id == SAFE_EOS_ID:
            continue
        piece = tok.decode([token_id], skip_special_tokens=True)
        if re.search(r"(=|\+|\-|\*|/|×|%|\\frac|\\sqrt)", piece):
            for j in range(max(0, idx - 2), min(len(weights), idx + 3)):
                weights[j] = max(weights[j], float(COMPUTATION_WEIGHT))

    answer_line = final_answer_tail(target_text)
    if answer_line:
        answer_ids = tok("\n" + answer_line, add_special_tokens=False)["input_ids"]
        answer_ids = ensure_eos(clamp_token_ids(answer_ids, len(tok)), SAFE_EOS_ID, len(target_ids))
        start = find_last_subsequence(target_ids, answer_ids)
        if start is None and answer_ids and answer_ids[0] != SAFE_EOS_ID:
            start = find_last_subsequence(target_ids, answer_ids[1:])
            if start is not None:
                answer_ids = answer_ids[1:]
        if start is not None:
            for j in range(start, min(len(weights), start + len(answer_ids))):
                weights[j] = max(weights[j], float(ANSWER_WEIGHT))
        else:
            tail = min(len(answer_ids), len(weights))
            for j in range(len(weights) - tail, len(weights)):
                weights[j] = max(weights[j], float(ANSWER_WEIGHT))

    return weights


def tokenize_sft_example(
    rec: dict,
    tok: Any,
    max_length: int,
    train_style: str,
    prompt_style: str,
    filter_no_answer: bool,
    filter_non_numeric_gold: bool,
    loss_style: str,
) -> Optional[dict[str, list[int]]]:
    prompt = build_grpo_prompt(rec) if uses_grpo_tagged_target(train_style) else build_prompt(rec, prompt_style)
    target = build_target(rec, train_style, filter_no_answer, filter_non_numeric_gold)
    if target is None:
        return None

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids_raw = tok("\n" + target, add_special_tokens=False)["input_ids"]
    target_ids_raw = ensure_eos(target_ids_raw, SAFE_EOS_ID, max_length)

    vocab_size = len(tok)
    prompt_ids = clamp_token_ids(prompt_ids, vocab_size)
    target_ids_raw = clamp_token_ids(target_ids_raw, vocab_size)

    if len(prompt_ids) + len(target_ids_raw) <= max_length:
        input_ids = prompt_ids + target_ids_raw
        target_labels = make_target_labels(target_ids_raw, target, tok, vocab_size, loss_style)
        labels = [-100] * len(prompt_ids) + target_labels
        item = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
        if loss_style in {"answer_weighted", "light_answer_weighted", "full_target_or_light_weighted"}:
            loss_weights = [0.0] * len(prompt_ids) + build_target_loss_weights(target_ids_raw, target, tok)
            prompt_end = min(len(prompt_ids), len(input_ids))
            loss_weights[:prompt_end] = [0.0] * prompt_end
            loss_weights = loss_weights[:len(input_ids)]
            assert len(input_ids) == len(labels) == len(loss_weights)
            item["loss_weights"] = loss_weights
        return item

    # Reserve room for a target tail so the model still sees the answer format.
    min_target_room = min(96, max(16, max_length // 4))
    if len(prompt_ids) > max_length - min_target_room:
        prompt_ids = prompt_ids[-(max_length - min_target_room):]

    target_budget = max_length - len(prompt_ids)
    if target_budget <= 0:
        return None

    answer_line = final_answer_tail(target)
    if answer_line:
        tail_ids = tok("\n" + answer_line, add_special_tokens=False)["input_ids"]
        tail_ids = ensure_eos(clamp_token_ids(tail_ids, vocab_size), SAFE_EOS_ID, target_budget)
        body_text = remove_final_answer_tail(target)
        body_ids = tok("\n" + body_text, add_special_tokens=False)["input_ids"] if body_text else []
        body_ids = clamp_token_ids(body_ids, vocab_size)

        if len(tail_ids) >= target_budget:
            target_ids = ensure_eos(tail_ids[:target_budget], SAFE_EOS_ID, target_budget)
        else:
            body_budget = target_budget - len(tail_ids)
            target_ids = body_ids[:body_budget] + tail_ids
    else:
        target_ids = ensure_eos(target_ids_raw[:target_budget], SAFE_EOS_ID, target_budget)

    input_ids = prompt_ids + target_ids
    target_labels = make_target_labels(target_ids, target, tok, vocab_size, loss_style)
    labels = [-100] * len(prompt_ids) + target_labels
    item = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    if loss_style in {"answer_weighted", "light_answer_weighted", "full_target_or_light_weighted"}:
        loss_weights = [0.0] * len(prompt_ids) + build_target_loss_weights(target_ids, target, tok)
        prompt_end = min(len(prompt_ids), len(input_ids))
        loss_weights[:prompt_end] = [0.0] * prompt_end
        loss_weights = loss_weights[:len(input_ids)]
        assert len(input_ids) == len(labels) == len(loss_weights)
        item["loss_weights"] = loss_weights
    return item


class SFTDataset(Dataset):
    """Tokenize prompt/target examples and mask loss on prompt + padding."""

    def __init__(
        self,
        records: list[dict],
        tok: Any,
        max_length: int,
        train_style: str = TRAIN_STYLE,
        prompt_style: str = PROMPT_STYLE,
        filter_no_answer: bool = FILTER_NO_ANSWER,
        filter_non_numeric_gold: bool = FILTER_NON_NUMERIC_GOLD,
        loss_style: str = LOSS_STYLE,
    ) -> None:
        self.records = []
        self.tok = tok
        self.max_length = max_length
        self.train_style = train_style
        self.prompt_style = prompt_style
        self.filter_no_answer = filter_no_answer
        self.filter_non_numeric_gold = filter_non_numeric_gold
        self.loss_style = loss_style

        skipped = 0
        for rec in records:
            if build_target(rec, train_style, filter_no_answer, filter_non_numeric_gold) is None:
                skipped += 1
            else:
                self.records.append(rec)

        print(
            f"SFTDataset: kept={len(self.records)} skipped={skipped} "
            f"train_style={train_style} prompt_style={prompt_style} "
            f"loss_style={loss_style} filter_non_numeric_gold={filter_non_numeric_gold}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        item = tokenize_sft_example(
            self.records[i],
            self.tok,
            self.max_length,
            self.train_style,
            self.prompt_style,
            self.filter_no_answer,
            self.filter_non_numeric_gold,
            self.loss_style,
        )
        if item is None:
            raise IndexError("Filtered record reached __getitem__; dataset construction should prevent this.")
        return item


@dataclass
class PadCollator:
    pad_id: int = SAFE_EOS_ID

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(x["input_ids"]) for x in batch)
        has_weights = any("loss_weights" in x for x in batch)
        out: dict[str, list[list[int] | list[float]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        if has_weights:
            out["loss_weights"] = []
        for x in batch:
            n = len(x["input_ids"])
            pad = maxlen - n
            out["input_ids"].append(x["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(x["attention_mask"] + [0] * pad)
            out["labels"].append(x["labels"] + [-100] * pad)
            if has_weights:
                weights = x.get("loss_weights", [0.0] * n)
                out["loss_weights"].append(weights + [0.0] * pad)

        tensors = {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
        }
        if has_weights:
            tensors["loss_weights"] = torch.tensor(out["loss_weights"], dtype=torch.float32)
        return tensors


class WeightedLossTrainer(Trainer):
    """Trainer variant that applies per-token weights to supervised target loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if loss_weights is None or labels is None:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].to(shift_logits.device).contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)
        active = (shift_labels != -100).to(per_token_loss.dtype) * shift_weights.to(per_token_loss.dtype)
        loss = (per_token_loss * active).sum() / active.sum().clamp_min(1e-8)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite weighted SFT loss: {loss}")
        return (loss, outputs) if return_outputs else loss


def preview_sft_examples(ds: SFTDataset, tok: Any, n: int = 2) -> None:
    for i in range(min(n, len(ds))):
        rec = ds.records[i]
        item = ds[i]
        input_text = tok.decode(item["input_ids"], skip_special_tokens=True)
        label_ids = [x for x in item["labels"] if x != -100]
        target_text = tok.decode(label_ids, skip_special_tokens=True)
        built_target = build_target(
            rec,
            ds.train_style,
            ds.filter_no_answer,
            ds.filter_non_numeric_gold,
        )
        print("=" * 100)
        print("IDX:", i)
        print("input_len:", len(item["input_ids"]), "| label_tokens:", len(label_ids))
        print("prompt labels masked:", all(x == -100 for x in item["labels"][: len(item["input_ids"]) - len(label_ids)]))
        print("EOS present:", item["input_ids"][-1] == SAFE_EOS_ID)
        print("DECODED INPUT:")
        print(input_text[:1200])
        if uses_grpo_tagged_target(ds.train_style) or ds.train_style in {"fdd_pot", "fdd_pot_compact"}:
            print("BUILT SPECIAL TARGET:")
            print((built_target or "")[:1200])
        print("DECODED TARGET:")
        print(target_text[:800])


train_ds_preview = SFTDataset(
    train_records[: min(3, len(train_records))],
    tokenizer,
    MAX_LENGTH,
    TRAIN_STYLE,
    PROMPT_STYLE,
    FILTER_NO_ANSWER,
    FILTER_NON_NUMERIC_GOLD,
    LOSS_STYLE,
)
preview_sft_examples(train_ds_preview, tokenizer, n=2)
del train_ds_preview

# %% [markdown]
# ## 6. Generation Function

# %%
def has_answer_marker(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in ANSWER_MARKER_PATTERNS)


def normalize_prediction_answer_line(line: str) -> str:
    ans = extract_answer(line)
    if ans:
        return f"{ANSWER_ANCHOR} {clean_answer_value_for_output(ans)}"
    return line.strip()


LAST_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*[-+]?\d+(?:[.,]\d+)?)?(?![\w])"
)


def extract_last_numeric_candidate(text: str) -> Optional[str]:
    matches = LAST_NUMBER_RE.findall(text or "")
    if not matches:
        return None
    for cand in reversed(matches):
        cand = cand.strip().strip("\"'“”‘’`")
        if parse_number(cand) is not None:
            return cand
    return None


def uses_fdd_pot_mode() -> bool:
    return TRAIN_STYLE in {"fdd_pot", "fdd_pot_compact"} or PROMPT_STYLE == "pot_instruction" or TRAIN_STAGE == "fdd_pot"


def extract_python_code(text: str) -> str:
    """Extract candidate model-generated Python from a raw completion."""
    source = text or ""
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", source, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        source = fence.group(1)
    if "Đáp án" in source:
        source = source.split("Đáp án", 1)[0]
    lines = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("Câu hỏi", "Bài giải", "Lời giải")):
            continue
        lines.append(line)
        if len(lines) >= FDD_PROGRAM_MAX_LINES:
            break
    return "\n".join(lines).strip()


ALLOWED_POT_AST_NODES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)
ALLOWED_POT_CALLS = {"print", "sqrt", "abs", "round", "min", "max"}


def validate_pot_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_POT_AST_NODES):
            raise ValueError(f"Disallowed Python node in generated PoT: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_POT_CALLS:
                raise ValueError("Disallowed function call in generated PoT.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Disallowed private name in generated PoT.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, type(None))):
            raise ValueError("Disallowed constant in generated PoT.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                if abs(float(node.right.value)) > 8:
                    raise ValueError("Exponent too large in generated PoT.")
            else:
                raise ValueError("Dynamic exponent disallowed in generated PoT.")


def safe_execute_pot(code: str) -> Optional[str]:
    """Execute a tiny arithmetic-only model-generated program and return answer."""
    code = extract_python_code(code)
    if not code or len(code) > 2500:
        return None
    try:
        tree = ast.parse(code, mode="exec")
        validate_pot_ast(tree)
    except Exception:
        return None

    printed: list[Any] = []

    def safe_print(*args: Any) -> None:
        if args:
            printed.append(args[-1])

    env: dict[str, Any] = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "print": safe_print,
        "pi": math.pi,
    }
    try:
        exec(compile(tree, "<generated_pot>", "exec"), env, env)
    except Exception:
        return None

    candidates = []
    if "answer" in env:
        candidates.append(str(env["answer"]))
    candidates.extend(str(x) for x in printed)
    for cand in reversed(candidates):
        if parse_number(cand) is not None:
            return cleanup_answer_string(cand)
    return None


def convert_pot_to_vietnamese_output(text: str) -> Optional[str]:
    if not FDD_EXECUTE_POT_AT_INFERENCE:
        return None
    code = extract_python_code(text)
    answer = safe_execute_pot(code)
    if answer is None:
        return None
    if FDD_KEEP_CODE_IN_OUTPUT and code:
        return normalize_space(f"Lời giải:\n{code}\nĐáp án là: {answer}")
    return f"Đáp án là: {answer}"


def has_equation_signal(text: str) -> bool:
    return bool(re.search(r"\d", text or "") and EQUATION_OPERATOR_RE.search(text or ""))


def numeric_values_in_text(text: str) -> list[float]:
    values = []
    for cand in LAST_NUMBER_RE.findall(text or ""):
        val = parse_number(cand.strip())
        if val is not None:
            values.append(val)
    return values


def malformed_tag_output(text: str) -> bool:
    text = text or ""
    think_matches = list(TAG_THINK_RE.finditer(text))
    answer_matches = list(TAG_ANSWER_RE.finditer(text))
    if has_repeated_grpo_tags(text):
        return True
    if len(think_matches) != 1 or len(answer_matches) != 1:
        return True
    return not (think_matches[0].start() < think_matches[0].end() <= answer_matches[0].start())


def reasoning_diagnostics_from_raw(raw_texts: list[str]) -> dict[str, float | int]:
    n = len(raw_texts)
    has_equation = 0
    repeated_think = 0
    answer_number_in_think = 0
    malformed = 0

    for text in raw_texts:
        lower = (text or "").lower()
        think_blocks = TAG_THINK_RE.findall(text or "")
        think_text = "\n".join(think_blocks)
        answer_text = extract_tagged_answer(text or "")
        answer_num = parse_number(answer_text) if answer_text is not None else None

        has_equation += int(any(has_equation_signal(block) for block in think_blocks))
        repeated_think += int(lower.count("<think>") > 1 or lower.count("</think>") > 1)
        malformed += int(malformed_tag_output(text or ""))

        if answer_num is not None and think_text:
            nums = numeric_values_in_text(think_text)
            answer_number_in_think += int(any((rel_error(num, answer_num) or 1.0) <= 0.01 for num in nums))

    return {
        "n": n,
        "has_equation_in_think_rate": has_equation / n if n else 0.0,
        "repeated_think_rate": repeated_think / n if n else 0.0,
        "answer_number_in_think_rate": answer_number_in_think / n if n else 0.0,
        "malformed_tag_rate": malformed / n if n else 0.0,
    }


def retrieval_tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+|\d+(?:[.,]\d+)?", normalize_space(text).lower(), flags=re.UNICODE))


def build_retrieval_index(records: list[dict]) -> list[tuple[dict, set[str]]]:
    return [(rec, retrieval_tokens(get_query_text(rec))) for rec in records]


def retrieve_train_examples(query: str, index: list[tuple[dict, set[str]]], k: int) -> list[dict]:
    if not index or k <= 0:
        return []
    q_tokens = retrieval_tokens(query)
    if not q_tokens:
        return []

    scored = []
    q_nums = {t for t in q_tokens if re.search(r"\d", t)}
    for rec, toks in index:
        overlap = len(q_tokens & toks)
        num_overlap = len(q_nums & toks)
        denom = math.sqrt(max(1, len(q_tokens)) * max(1, len(toks)))
        score = (overlap + 2.0 * num_overlap) / denom
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in scored[:k]]


def generation_settings(train_style: str, decoding_style: str, requested_max_new_tokens: int) -> dict[str, Any]:
    max_new = requested_max_new_tokens
    settings: dict[str, Any] = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new,
    }

    if decoding_style == "greedy":
        return settings
    if decoding_style == "beam2":
        settings["num_beams"] = 2
        settings["early_stopping"] = True
        return settings
    if decoding_style == "short_greedy":
        if train_style in {"answer_stub", "answer_only", "answer_focused"}:
            max_new = min(max_new, 32)
        elif train_style == "equation_focused":
            max_new = min(max_new, 64)
        elif train_style in {"tagged_short_solution", "tagged_equation_focused"}:
            max_new = min(max_new, 128)
        elif train_style == "tagged_equation_minimal":
            max_new = min(max_new, 80)
        elif train_style == "tagged_answer_only":
            max_new = min(max_new, 64)
        elif train_style == "fdd_pot":
            max_new = min(max_new, 160)
        elif train_style == "fdd_pot_compact":
            max_new = min(max_new, 96)
        elif train_style == "fdd_lite_type_routing":
            max_new = min(max_new, 48)
        else:
            max_new = min(max_new, 96)
        settings["max_new_tokens"] = max_new
        return settings

    raise ValueError(f"Unknown DECODING_STYLE: {decoding_style}")


def clean_answer_value_for_output(answer_text: str) -> str:
    ans = cleanup_answer_string(answer_text)
    ans = re.sub(r"(?<=\d)(?:hue)+$", "", ans, flags=re.IGNORECASE).strip()

    # Preserve common exact symbolic scalar answers.
    symbolic_patterns = [
        r"^[+-]?\d+\\(?:d|t)?frac\{[^{}]+\}\{[^{}]+\}",
        r"^\\(?:d|t)?frac\{[^{}]+\}\{[^{}]+\}",
        r"^[+-]?\d+(?:[.,]\d+)?\\sqrt\{[^{}]+\}",
        r"^[+-]?\d+(?:[.,]\d+)?\\sqrt\s*\d+",
        r"^\\sqrt\{[^{}]+\}",
    ]
    for pat in symbolic_patterns:
        m = re.match(pat, ans)
        if m:
            return m.group(0).strip()

    # Strip units after scalar numeric answers.
    numeric_prefix = re.match(
        r"^\s*([+-]?\d+(?:[.,]\d+)?(?:\s*/\s*[+-]?\d+(?:[.,]\d+)?)?)\b",
        ans,
    )
    if numeric_prefix and parse_number(numeric_prefix.group(1)) is not None:
        return numeric_prefix.group(1).strip()

    if parse_number(ans) is not None:
        return ans

    cand = extract_last_numeric_candidate(ans)
    return cand or ans


def clean_generated_answer(text: str) -> str:
    """
    Return final model output in the form:
    ####đáp án là: <clean_answer>
    """
    answer = extract_answer(text)
    if answer is None:
        tagged = extract_tagged_answer(text or "")
        answer = tagged if tagged is not None else extract_last_numeric_candidate(text or "")
    answer = clean_answer_value_for_output(answer or "")
    return normalize_space(f"{ANSWER_ANCHOR} {answer}").rstrip()


def uses_answer_anchor_output() -> bool:
    return TRAIN_STYLE == "fdd_lite_type_routing" or LOSS_STYLE == "answer_weighted"


def postprocess_generation(text: str, prompt: Optional[str] = None) -> str:
    text = normalize_space(text or "")
    text = strip_trailing_special_artifacts(text)

    if prompt:
        prompt_clean = normalize_space(prompt)
        if text.startswith(prompt_clean):
            text = normalize_space(text[len(prompt_clean):])

    # If an answer marker appears, keep text only through the first final answer line.
    marker_matches = []
    for pat in ANSWER_MARKER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            marker_matches.append((m.start(), m.end()))

    if marker_matches:
        start, _ = sorted(marker_matches, key=lambda x: x[0])[0]
        before = normalize_space(text[:start])
        after = text[start:].split("\n")[0].strip()
        answer_line = normalize_prediction_answer_line(after)
        text = normalize_space(f"{before}\n{answer_line}" if before else answer_line)

    # Remove duplicate final answer lines after normalization.
    lines = []
    seen_answer_line = False
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if has_answer_marker(line):
            if seen_answer_line:
                continue
            line = normalize_prediction_answer_line(line)
            seen_answer_line = True
        line = strip_trailing_special_artifacts(line)
        lines.append(line)
    text = "\n".join(lines).strip()

    if not text:
        text = ANSWER_ANCHOR
    elif POSTPROCESS_APPEND_LAST_NUMBER and extract_answer(text) is None:
        cand = extract_last_numeric_candidate(text)
        if cand is not None:
            text = normalize_space(f"{text}\n{ANSWER_ANCHOR} {clean_answer_value_for_output(cand)}")
    return text


def generate_outputs(
    model_path_or_name: str | Path,
    records: list[dict],
    output_path: str | Path,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = GEN_BATCH_SIZE,
    retrieval_records: Optional[list[dict]] = None,
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_path_or_name} on {device} ...", flush=True)

    gen_tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(model_path_or_name, local_files_only=True)
    )
    gen_tokenizer.padding_side = "left"

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = {"local_files_only": True}
    # Newer Transformers prefers dtype; older versions use torch_dtype.
    if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path_or_name, **model_kwargs).to(device)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.eval()

    vocab_n = model.transformer.wte.num_embeddings
    outputs: list[dict] = []
    raw_generation_texts: list[str] = []
    t0 = time.time()
    retrieval_index = build_retrieval_index(retrieval_records or []) if USE_TRAIN_RETRIEVAL_CONTEXT else []
    gen_settings = generation_settings(TRAIN_STYLE, DECODING_STYLE, max_new_tokens)
    print("DECODING_STYLE:", DECODING_STYLE, "| generation settings:", gen_settings)
    if USE_TRAIN_RETRIEVAL_CONTEXT:
        print(
            "USE_TRAIN_RETRIEVAL_CONTEXT=True. Use only if instructor permits train-set few-shot retrieval. "
            f"retrieval_records={len(retrieval_records or [])} | k={N_RETRIEVED_EXAMPLES}"
        )

    with torch.inference_mode():
        for start in tqdm(range(0, len(records), batch_size), desc="Generating"):
            batch_records = records[start:start + batch_size]
            prompts = []
            for rec in batch_records:
                retrieved = retrieve_train_examples(
                    get_query_text(rec),
                    retrieval_index,
                    N_RETRIEVED_EXAMPLES,
                ) if retrieval_index else None
                prompts.append(build_inference_prompt(rec, retrieved))
            enc = gen_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)

            ids = enc["input_ids"].clamp(max=vocab_n - 1)
            attention_mask = enc.get("attention_mask")

            gen = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                **gen_settings,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=SAFE_EOS_ID,
                eos_token_id=SAFE_EOS_ID,
            )

            prompt_width = ids.shape[1]
            for j, rec in enumerate(batch_records):
                raw = gen_tokenizer.decode(gen[j, prompt_width:], skip_special_tokens=True)
                raw_generation_texts.append(raw)
                pot_converted = convert_pot_to_vietnamese_output(raw) if uses_fdd_pot_mode() else None
                tagged_converted = convert_tagged_to_vietnamese_output(raw) if GRPO_USE_THINK_ANSWER_TAGS else None
                anchor_converted = clean_generated_answer(raw) if uses_answer_anchor_output() else None
                cleaned = pot_converted or tagged_converted or anchor_converted or postprocess_generation(raw, prompts[j])
                outputs.append({
                    "id": len(outputs),
                    "query_vi": get_query_text(rec),
                    "type": rec.get("type"),
                    "model_output": cleaned,
                })

    save_json(outputs, output_path)
    out_hash = sha256_file(output_path)
    Path(str(output_path) + ".sha256.txt").write_text(out_hash + "\n", encoding="utf-8")

    dt = time.time() - t0
    print(f"Wrote {output_path} | {dt / 60:.2f} min | SHA256: {out_hash}")
    if GRPO_USE_THINK_ANSWER_TAGS and raw_generation_texts and (
        uses_grpo_tagged_target(TRAIN_STYLE) or TRAIN_STAGE in {"grpo", "sft_then_grpo"}
    ):
        diag = reasoning_diagnostics_from_raw(raw_generation_texts)
        print("Reasoning/tag diagnostics:")
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        update_experiment_report(
            last_reasoning_diagnostics=diag,
            **{f"{Path(output_path).stem}_reasoning_diagnostics": diag},
        )

    del model
    torch.cuda.empty_cache()
    return outputs


def generate_outputs_with_loaded_model(
    model: Any,
    gen_tokenizer: Any,
    records: list[dict],
    output_path: str | Path,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = GEN_BATCH_SIZE,
    retrieval_records: Optional[list[dict]] = None,
) -> list[dict]:
    device = next(model.parameters()).device
    gen_tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()

    vocab_n = model.transformer.wte.num_embeddings
    outputs: list[dict] = []
    raw_generation_texts: list[str] = []
    retrieval_index = build_retrieval_index(retrieval_records or []) if USE_TRAIN_RETRIEVAL_CONTEXT else []
    gen_settings = generation_settings(TRAIN_STYLE, DECODING_STYLE, max_new_tokens)

    with torch.inference_mode():
        for start in tqdm(range(0, len(records), batch_size), desc="Generating loaded model"):
            batch_records = records[start:start + batch_size]
            prompts = []
            for rec in batch_records:
                retrieved = retrieve_train_examples(
                    get_query_text(rec),
                    retrieval_index,
                    N_RETRIEVED_EXAMPLES,
                ) if retrieval_index else None
                prompts.append(build_inference_prompt(rec, retrieved))
            enc = gen_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)
            ids = enc["input_ids"].clamp(max=vocab_n - 1)
            gen = model.generate(
                input_ids=ids,
                attention_mask=enc.get("attention_mask"),
                **gen_settings,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                pad_token_id=SAFE_EOS_ID,
                eos_token_id=SAFE_EOS_ID,
            )
            prompt_width = ids.shape[1]
            for j, rec in enumerate(batch_records):
                raw = gen_tokenizer.decode(gen[j, prompt_width:], skip_special_tokens=True)
                raw_generation_texts.append(raw)
                pot_converted = convert_pot_to_vietnamese_output(raw) if uses_fdd_pot_mode() else None
                tagged_converted = convert_tagged_to_vietnamese_output(raw) if GRPO_USE_THINK_ANSWER_TAGS else None
                anchor_converted = clean_generated_answer(raw) if uses_answer_anchor_output() else None
                cleaned = pot_converted or tagged_converted or anchor_converted or postprocess_generation(raw, prompts[j])
                outputs.append({
                    "id": len(outputs),
                    "query_vi": get_query_text(rec),
                    "type": rec.get("type"),
                    "model_output": cleaned,
                })

    save_json(outputs, output_path)
    out_hash = sha256_file(output_path)
    Path(str(output_path) + ".sha256.txt").write_text(out_hash + "\n", encoding="utf-8")
    print(f"Wrote {output_path} | SHA256: {out_hash}")
    if GRPO_USE_THINK_ANSWER_TAGS and raw_generation_texts and (
        uses_grpo_tagged_target(TRAIN_STYLE) or TRAIN_STAGE in {"grpo", "sft_then_grpo"}
    ):
        diag = reasoning_diagnostics_from_raw(raw_generation_texts)
        print("Reasoning/tag diagnostics:")
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        update_experiment_report(
            last_reasoning_diagnostics=diag,
            **{f"{Path(output_path).stem}_reasoning_diagnostics": diag},
        )
    if was_training:
        model.train()
    return outputs


def voting_generation_specs() -> list[dict[str, Any]]:
    specs = [
        {"name": "greedy", "do_sample": False, "num_beams": 1},
        {"name": "low_temperature", "do_sample": True, "temperature": 0.2, "top_p": 0.9, "num_beams": 1},
        {"name": "medium_temperature", "do_sample": True, "temperature": 0.4, "top_p": 0.9, "num_beams": 1},
    ]
    return specs[: max(1, NUM_CANDIDATES)]


def normalized_candidate_answer(text: str) -> str:
    answer = extract_answer(clean_generated_answer(text))
    return clean_answer_value_for_output(answer or "")


def choose_voted_answer(candidate_answers: list[str]) -> tuple[str, str]:
    greedy_answer = candidate_answers[0] if candidate_answers else ""
    counts = Counter(a for a in candidate_answers if a)
    if counts:
        answer, count = counts.most_common(1)[0]
        if count >= 2:
            return answer, f"majority_{count}_of_{len(candidate_answers)}"
    return greedy_answer, "fallback_greedy"


def generate_voting_outputs(
    model_path_or_name: str | Path,
    records: list[dict],
    output_path: str | Path,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = GEN_BATCH_SIZE,
    include_debug_metadata: bool = False,
    greedy_output_path: Optional[str | Path] = None,
) -> tuple[list[dict], list[dict]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_path_or_name} for inference voting on {device} ...", flush=True)

    gen_tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(model_path_or_name, local_files_only=True)
    )
    gen_tokenizer.padding_side = "left"

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = {"local_files_only": True}
    if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_path_or_name, **model_kwargs).to(device)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.eval()

    specs = voting_generation_specs()
    print("INFERENCE_VOTING=True | specs:", specs)
    vocab_n = model.transformer.wte.num_embeddings
    outputs: list[dict] = []
    greedy_outputs: list[dict] = []
    t0 = time.time()

    with torch.inference_mode():
        for start in tqdm(range(0, len(records), batch_size), desc="Voting generation"):
            batch_records = records[start:start + batch_size]
            prompts = [build_inference_prompt(rec, None) for rec in batch_records]
            enc = gen_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)
            ids = enc["input_ids"].clamp(max=vocab_n - 1)
            attention_mask = enc.get("attention_mask")
            prompt_width = ids.shape[1]

            batch_candidates: list[list[str]] = [[] for _ in batch_records]
            for spec_idx, spec in enumerate(specs):
                torch.manual_seed(SEED + spec_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED + spec_idx)
                gen_kwargs = {
                    "input_ids": ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": spec["do_sample"],
                    "num_beams": spec.get("num_beams", 1),
                    "repetition_penalty": REPETITION_PENALTY,
                    "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
                    "pad_token_id": SAFE_EOS_ID,
                    "eos_token_id": SAFE_EOS_ID,
                }
                if spec["do_sample"]:
                    gen_kwargs["temperature"] = spec["temperature"]
                    gen_kwargs["top_p"] = spec["top_p"]
                gen = model.generate(**gen_kwargs)
                for j in range(len(batch_records)):
                    raw = gen_tokenizer.decode(gen[j, prompt_width:], skip_special_tokens=True)
                    batch_candidates[j].append(raw)

            for rec, candidates in zip(batch_records, batch_candidates):
                candidate_outputs = [clean_generated_answer(x) for x in candidates]
                candidate_answers = [normalized_candidate_answer(x) for x in candidate_outputs]
                selected_answer, reason = choose_voted_answer(candidate_answers)
                if not selected_answer:
                    selected_answer = candidate_answers[0] if candidate_answers else ""
                model_output = normalize_space(f"{ANSWER_ANCHOR} {selected_answer}")

                item = {
                    "id": len(outputs),
                    "query_vi": get_query_text(rec),
                    "type": rec.get("type"),
                    "model_output": model_output,
                }
                if include_debug_metadata:
                    item.update({
                        "candidates": candidate_outputs,
                        "candidate_answers": candidate_answers,
                        "selected_answer": selected_answer,
                        "selection_reason": reason,
                    })
                outputs.append(item)
                greedy_outputs.append({
                    "id": len(greedy_outputs),
                    "query_vi": get_query_text(rec),
                    "type": rec.get("type"),
                    "model_output": candidate_outputs[0] if candidate_outputs else model_output,
                })

    save_json(outputs, output_path)
    if greedy_output_path is not None:
        save_json(greedy_outputs, greedy_output_path)
    out_hash = sha256_file(output_path)
    Path(str(output_path) + ".sha256.txt").write_text(out_hash + "\n", encoding="utf-8")
    dt = time.time() - t0
    print(f"Wrote {output_path} | voting generation {dt / 60:.2f} min | SHA256: {out_hash}")

    del model
    torch.cuda.empty_cache()
    return outputs, greedy_outputs


def run_pretrain_smoke_checks() -> None:
    assert len(train_records) > 0
    assert len(valid_records) > 0
    assert "bay" not in _SIMPLE

    if TRAIN_STYLE in {
        "fdd_lite_type_routing",
        "fdd_pot_compact",
        "type_specific_equation_targets",
        "original_fdd_pot_restored",
    } and RUN_TRAIN:
        sample_targets = train_records[: min(5000, len(train_records))]
        assert all(count_answer_anchors(str(r.get("_target", ""))) == 1 for r in sample_targets)

    assert clean_generated_answer("####đáp án là: 72 ô") == "####đáp án là: 72"
    assert clean_generated_answer("####đáp án là: 5huehue") == "####đáp án là: 5"
    assert clean_generated_answer("####đáp án là: \\frac{1}{2}") == "####đáp án là: \\frac{1}{2}"
    assert clean_generated_answer("####đáp án là: 50\\sqrt{10}") == "####đáp án là: 50\\sqrt{10}"
    assert clean_generated_answer("####đáp án là: 14\\frac{6}{7}") == "####đáp án là: 14\\frac{6}{7}"

    smoke_ds = SFTDataset(
        train_records[: min(16, len(train_records))],
        tokenizer,
        MAX_LENGTH,
        TRAIN_STYLE,
        PROMPT_STYLE,
        FILTER_NO_ANSWER,
        FILTER_NON_NUMERIC_GOLD,
        LOSS_STYLE,
    )
    assert len(smoke_ds) > 0
    item = smoke_ds[0]
    assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
    if "loss_weights" in item:
        assert len(item["input_ids"]) == len(item["loss_weights"])
    batch = PadCollator(pad_id=SAFE_EOS_ID)([item])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    if "loss_weights" in batch:
        assert batch["input_ids"].shape == batch["loss_weights"].shape
    print("Smoke checks passed: target anchors, cleanup, dataset item, and collated batch are sane.")


run_pretrain_smoke_checks()


# %% [markdown]
# ## 6b. Optional GRPO / RLVR Utilities

# %%
def model_dir_has_weights(path: Path) -> bool:
    return path.exists() and (any(path.glob("*.bin")) or any(path.glob("*.safetensors")))


def resolve_grpo_start_checkpoint() -> str | Path:
    if SFT_CKPT_DIR:
        p = Path(SFT_CKPT_DIR)
        if p.exists():
            print("GRPO start checkpoint from SFT_CKPT_DIR:", p)
            return p
        print("SFT_CKPT_DIR was set but does not exist:", p)

    if TRAIN_STAGE == "sft_then_grpo" and OUTPUT_DIR.exists():
        print("GRPO start checkpoint from OUTPUT_DIR after SFT:", OUTPUT_DIR)
        return OUTPUT_DIR

    if OUTPUT_DIR.exists() and model_dir_has_weights(OUTPUT_DIR):
        print("GRPO start checkpoint from existing OUTPUT_DIR:", OUTPUT_DIR)
        return OUTPUT_DIR

    print("GRPO start checkpoint from base MODEL_NAME:", MODEL_NAME)
    return MODEL_NAME


def resolve_sft_start_checkpoint() -> str | Path:
    if SFT_CKPT_DIR:
        p = Path(SFT_CKPT_DIR)
        if p.exists() and model_dir_has_weights(p):
            print("SFT start checkpoint from SFT_CKPT_DIR:", p)
            return p
        raise FileNotFoundError(f"SFT_CKPT_DIR was set but is not a valid model checkpoint: {p}")

    print("SFT start checkpoint from base MODEL_NAME:", MODEL_NAME)
    return MODEL_NAME


def resolve_generation_checkpoint() -> str | Path:
    if SFT_CKPT_DIR:
        p = Path(SFT_CKPT_DIR)
        if p.exists() and model_dir_has_weights(p):
            print("Generation checkpoint from SFT_CKPT_DIR:", p)
            return p
        print("SFT_CKPT_DIR was set but is not a valid model checkpoint:", p)

    if INFERENCE_VOTING and OUTPUT_DIR.exists() and model_dir_has_weights(OUTPUT_DIR):
        print("Generation checkpoint from OUTPUT_DIR:", OUTPUT_DIR)
        return OUTPUT_DIR

    if INFERENCE_VOTING and not RUN_TRAIN and not ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING:
        raise FileNotFoundError(
            f"INFERENCE_VOTING=True expects an existing frozen checkpoint at {OUTPUT_DIR}. "
            "Set SFT_CKPT_DIR to a local checkpoint, train earlier in the same session, "
            "or set ALLOW_BASE_MODEL_FALLBACK_FOR_VOTING=True for a debug fallback."
        )

    if INFERENCE_VOTING and not RUN_TRAIN:
        print(
            "WARNING: INFERENCE_VOTING=True but no trained checkpoint was found at "
            f"{OUTPUT_DIR}. Falling back to the base model at {MODEL_NAME}. "
            "This is only for smoke/debug output and will not reflect the trained FDD checkpoint."
        )
        return Path(MODEL_NAME)

    if TRAIN_STAGE in {"grpo", "sft_then_grpo"} and GRPO_OUTPUT_DIR.exists() and model_dir_has_weights(GRPO_OUTPUT_DIR):
        return GRPO_OUTPUT_DIR
    if OUTPUT_DIR.exists() and model_dir_has_weights(OUTPUT_DIR):
        return OUTPUT_DIR
    return Path(MODEL_NAME)


def calculate_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_tokens = input_ids[:, 1:].unsqueeze(-1)
    gathered = log_probs.gather(-1, next_tokens).squeeze(-1)
    first = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=gathered.dtype)
    token_logprobs = torch.cat([first, gathered], dim=1)
    if attention_mask is not None:
        token_logprobs = token_logprobs * attention_mask.to(token_logprobs.dtype)
    return token_logprobs


def grpo_train_candidates(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        if not str(rec.get("query_vi", "")).strip() or not str(rec.get("response_vi", "")).strip():
            continue
        gold = extract_answer(rec.get("response_vi"))
        if gold is None or parse_number(gold) is None:
            continue
        out.append(rec)
    return out[:GRPO_MAX_TRAIN_SAMPLES]


def pad_grpo_sequences(items: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in items)
    out = {
        "input_ids": [],
        "attention_mask": [],
        "response_mask": [],
        "old_log_probs": [],
        "advantages": [],
    }
    for x in items:
        n = len(x["input_ids"])
        pad = max_len - n
        out["input_ids"].append(x["input_ids"] + [SAFE_EOS_ID] * pad)
        out["attention_mask"].append(x["attention_mask"] + [0] * pad)
        out["response_mask"].append(x["response_mask"] + [0.0] * pad)
        out["old_log_probs"].append(x["old_log_probs"] + [0.0] * pad)
        out["advantages"].append(float(x["advantage"]))
    return {
        "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
        "response_mask": torch.tensor(out["response_mask"], dtype=torch.float32),
        "old_log_probs": torch.tensor(out["old_log_probs"], dtype=torch.float32),
        "advantages": torch.tensor(out["advantages"], dtype=torch.float32),
    }


def first_eos_truncated(ids: list[int]) -> list[int]:
    if SAFE_EOS_ID in ids:
        return ids[: ids.index(SAFE_EOS_ID) + 1]
    return ids


def collect_grpo_rollouts(
    model: Any,
    tok: Any,
    records: list[dict],
    max_items: int,
) -> tuple[list[dict], dict]:
    assert records is train_records, "GRPO training must use the selected train.json records only."

    device = next(model.parameters()).device
    candidates = grpo_train_candidates(records)
    if not candidates:
        raise ValueError("No GRPO train candidates with numeric gold answers.")

    model.eval()
    tok.padding_side = "left"
    rng = random.Random(SEED)
    order = list(candidates)
    rng.shuffle(order)

    rollouts: list[dict] = []
    prompt_groups = 0
    no_signal_groups = 0
    stats_rewards = []
    stats_correct = []
    stats_format = []
    tagged_extractable = 0
    numeric_tagged_answers = 0
    repeated_tags = 0
    completion_lengths = []
    sample_debug = []
    raw_completion_texts = []

    print("GRPO prompt preview:")
    for rec in order[:2]:
        print("=" * 100)
        print(build_grpo_prompt(rec)[:1200])

    with torch.no_grad():
        for start in tqdm(range(0, len(order), GRPO_ROLLOUT_BATCH_SIZE), desc="GRPO rollouts"):
            batch_records = order[start:start + GRPO_ROLLOUT_BATCH_SIZE]
            prompts = [build_grpo_prompt(rec) for rec in batch_records]
            gold_answers = [extract_answer(rec.get("response_vi")) or "" for rec in batch_records]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)

            prompt_width = enc["input_ids"].shape[1]
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
            too_long = [n >= MAX_LENGTH for n in prompt_lens]
            if all(too_long):
                continue

            gen_kwargs = dict(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                do_sample=True,
                temperature=GRPO_TEMPERATURE,
                top_p=GRPO_TOP_P,
                max_new_tokens=GRPO_MAX_NEW_TOKENS,
                num_return_sequences=GRPO_NUM_GENERATIONS,
                pad_token_id=SAFE_EOS_ID,
                eos_token_id=SAFE_EOS_ID,
            )
            if GRPO_TOP_K and GRPO_TOP_K > 0:
                gen_kwargs["top_k"] = GRPO_TOP_K

            generated = model.generate(**gen_kwargs)
            batch_experiences = []

            for i, rec in enumerate(batch_records):
                if too_long[i]:
                    continue
                group_rewards = []
                group_items = []
                prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].detach().cpu().tolist()
                prompt_text = prompts[i]
                gold_answer = gold_answers[i]

                for g in range(GRPO_NUM_GENERATIONS):
                    row = i * GRPO_NUM_GENERATIONS + g
                    seq = generated[row].detach().cpu().tolist()
                    completion_ids = first_eos_truncated(seq[prompt_width:])
                    if not completion_ids:
                        completion_ids = [SAFE_EOS_ID]
                    completion_text = tok.decode(completion_ids, skip_special_tokens=True)
                    raw_completion_texts.append(completion_text)
                    reward = compute_grpo_reward(completion_text, gold_answer)

                    full_ids = prompt_ids + completion_ids
                    response_mask = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
                    item = {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "response_mask": response_mask,
                        "reward": reward,
                        "prompt_text": prompt_text,
                        "completion_text": completion_text,
                        "gold_answer": gold_answer,
                        "pred_answer": reward.get("pred_answer"),
                        "type": str(rec.get("type") or "unknown"),
                    }
                    group_items.append(item)
                    group_rewards.append(float(reward["total"]))
                    stats_rewards.append(float(reward["total"]))
                    stats_correct.append(float(reward["correctness"]))
                    stats_format.append(float(reward["format"]))
                    pred_answer = reward.get("pred_answer")
                    tagged_extractable += int(pred_answer is not None)
                    numeric_tagged_answers += int(pred_answer is not None and parse_number(str(pred_answer)) is not None)
                    repeated_tags += int(has_repeated_grpo_tags(completion_text))
                    completion_lengths.append(len(completion_ids))
                    if len(sample_debug) < 4:
                        sample_debug.append({
                            "type": item["type"],
                            "gold_answer": gold_answer,
                            "completion_text": completion_text[:1000],
                            "reward": reward,
                        })

                if not group_items:
                    continue
                prompt_groups += 1
                mean_r = sum(group_rewards) / len(group_rewards)
                var = sum((x - mean_r) ** 2 for x in group_rewards) / max(1, len(group_rewards))
                std = math.sqrt(var)
                if std <= GRPO_MIN_GROUP_REWARD_STD:
                    no_signal_groups += 1
                    advs = [0.0] * len(group_rewards)
                else:
                    advs = [(x - mean_r) / (std + GRPO_MIN_GROUP_REWARD_STD) for x in group_rewards]
                for item, adv in zip(group_items, advs):
                    item["advantage"] = float(adv)
                    batch_experiences.append(item)

            if batch_experiences:
                max_len = max(len(x["input_ids"]) for x in batch_experiences)
                old_input_ids = []
                old_attention_mask = []
                for x in batch_experiences:
                    n = len(x["input_ids"])
                    pad = max_len - n
                    old_input_ids.append(x["input_ids"] + [SAFE_EOS_ID] * pad)
                    old_attention_mask.append(x["attention_mask"] + [0] * pad)
                padded = {
                    "input_ids": torch.tensor(old_input_ids, dtype=torch.long, device=device),
                    "attention_mask": torch.tensor(old_attention_mask, dtype=torch.long, device=device),
                }
                old_log_probs = calculate_token_logprobs(
                    model,
                    padded["input_ids"],
                    padded["attention_mask"],
                ).detach().cpu()
                for item, old_lp in zip(batch_experiences, old_log_probs):
                    n = len(item["input_ids"])
                    item["old_log_probs"] = old_lp[:n].tolist()
                    rollouts.append(item)
                    if len(rollouts) >= max_items:
                        break

            if len(rollouts) >= max_items:
                break

    n_rollouts = len(rollouts)
    stats = {
        "rollout_examples": n_rollouts,
        "prompt_groups": prompt_groups,
        "mean_total_reward": sum(stats_rewards) / len(stats_rewards) if stats_rewards else 0.0,
        "mean_correctness_reward": sum(stats_correct) / len(stats_correct) if stats_correct else 0.0,
        "mean_format_reward": sum(stats_format) / len(stats_format) if stats_format else 0.0,
        "tagged_answer_extractable_rate": tagged_extractable / len(stats_rewards) if stats_rewards else 0.0,
        "numeric_tagged_answer_rate": numeric_tagged_answers / len(stats_rewards) if stats_rewards else 0.0,
        "exact_correct_rate": sum(1 for x in stats_correct if x >= 1.0) / len(stats_correct) if stats_correct else 0.0,
        "no_signal_group_rate": no_signal_groups / prompt_groups if prompt_groups else 0.0,
        "average_completion_token_length": sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0.0,
        "repeated_tag_rate": repeated_tags / len(stats_rewards) if stats_rewards else 0.0,
        "samples": sample_debug,
    }
    stats.update(reasoning_diagnostics_from_raw(raw_completion_texts))
    print("GRPO rollout stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:6000])
    return rollouts, stats


def grpo_preflight_decision(stats: dict) -> tuple[bool, str]:
    exact_correct_rate = float(stats.get("exact_correct_rate") or 0.0)
    mean_correctness = float(stats.get("mean_correctness_reward") or 0.0)
    tagged_rate = float(stats.get("tagged_answer_extractable_rate") or 0.0)

    reasons = []
    if exact_correct_rate == 0.0:
        reasons.append("exact_correct_rate == 0")
    if mean_correctness == 0.0:
        reasons.append("mean_correctness_reward == 0")
    if tagged_rate < 0.60:
        reasons.append(f"tagged_answer_extractable_rate < 0.60 ({tagged_rate:.3f})")
    if reasons:
        return False, "; ".join(reasons)
    return True, "rollout has sufficient correctness/format signal"


def save_grpo_preflight_diagnostics(iteration: int, stats: dict, passed: bool, reason: str) -> None:
    payload = {
        "iteration": iteration,
        "preflight_passed": passed,
        "preflight_reason": reason,
        "tagged_answer_extractable_rate": float(stats.get("tagged_answer_extractable_rate") or 0.0),
        "numeric_tagged_answer_rate": float(stats.get("numeric_tagged_answer_rate") or 0.0),
        "exact_correct_rate": float(stats.get("exact_correct_rate") or 0.0),
        "mean_correctness_reward": float(stats.get("mean_correctness_reward") or 0.0),
        "mean_format_reward": float(stats.get("mean_format_reward") or 0.0),
        "no_signal_group_rate": float(stats.get("no_signal_group_rate") or 0.0),
        "has_equation_in_think_rate": float(stats.get("has_equation_in_think_rate") or 0.0),
        "repeated_think_rate": float(stats.get("repeated_think_rate") or 0.0),
        "answer_number_in_think_rate": float(stats.get("answer_number_in_think_rate") or 0.0),
        "malformed_tag_rate": float(stats.get("malformed_tag_rate") or 0.0),
        "rollout_stats": stats,
    }
    save_json(payload, GRPO_PREFLIGHT_PATH)
    update_experiment_report(
        grpo_preflight=payload,
        preflight_passed=passed,
        preflight_reason=reason,
        tagged_answer_extractable_rate=payload["tagged_answer_extractable_rate"],
        numeric_tagged_answer_rate=payload["numeric_tagged_answer_rate"],
        exact_correct_rate=payload["exact_correct_rate"],
        mean_correctness_reward=payload["mean_correctness_reward"],
        mean_format_reward=payload["mean_format_reward"],
        no_signal_group_rate=payload["no_signal_group_rate"],
        has_equation_in_think_rate=payload["has_equation_in_think_rate"],
        repeated_think_rate=payload["repeated_think_rate"],
        answer_number_in_think_rate=payload["answer_number_in_think_rate"],
        malformed_tag_rate=payload["malformed_tag_rate"],
    )
    print("Wrote GRPO preflight diagnostics:", GRPO_PREFLIGHT_PATH)


def grpo_collate(items: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    batch = pad_grpo_sequences(items)
    return {k: v.to(device) for k, v in batch.items()}


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def grpo_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    old_log_probs = batch["old_log_probs"].detach()
    advantages = batch["advantages"].detach().unsqueeze(1)

    current_log_probs = calculate_token_logprobs(model, input_ids, attention_mask)
    assert current_log_probs.shape == old_log_probs.shape == response_mask.shape

    ratio = torch.exp(current_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - GRPO_CLIP_EPS, 1.0 + GRPO_CLIP_EPS) * advantages
    objective = torch.minimum(unclipped, clipped)
    loss = -masked_mean(objective, response_mask)

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite GRPO loss: {loss}")
    return loss


def save_grpo_checkpoint(model: Any, tok: Any) -> None:
    GRPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(GRPO_OUTPUT_DIR)
    tok.save_pretrained(GRPO_OUTPUT_DIR)
    (GRPO_OUTPUT_DIR / "model_hash.txt").write_text(sha256_dir(GRPO_OUTPUT_DIR) + "\n", encoding="utf-8")


def run_grpo_training(start_model_dir_or_name: str | Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting GRPO from:", start_model_dir_or_name)
    print("GRPO_USE_THINK_ANSWER_TAGS:", GRPO_USE_THINK_ANSWER_TAGS)

    grpo_tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(start_model_dir_or_name, local_files_only=True)
    )
    model = AutoModelForCausalLM.from_pretrained(start_model_dir_or_name, local_files_only=True).to(device)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=GRPO_LR)
    grpo_history = []

    for iteration in range(GRPO_ITERATIONS):
        print("=" * 100)
        print(f"GRPO iteration {iteration + 1}/{GRPO_ITERATIONS}")
        rollouts, rollout_stats = collect_grpo_rollouts(
            model,
            grpo_tokenizer,
            train_records,
            GRPO_BUFFER_SIZE,
        )
        if not rollouts:
            print("No GRPO rollouts collected; stopping.")
            break

        preflight_passed, preflight_reason = grpo_preflight_decision(rollout_stats)
        save_grpo_preflight_diagnostics(iteration + 1, rollout_stats, preflight_passed, preflight_reason)

        iter_summary = {
            "iteration": iteration + 1,
            "avg_loss": None,
            "rollout_stats": rollout_stats,
            "preflight_passed": preflight_passed,
            "preflight_reason": preflight_reason,
        }

        if not preflight_passed:
            print("GRPO skipped: rollout has insufficient correctness/format signal.")
            print("Preflight reason:", preflight_reason)
            grpo_history.append(iter_summary)
            update_experiment_report(grpo_history=grpo_history, grpo_last_checkpoint=str(resolve_generation_checkpoint()))
            break

        if GRPO_PREFLIGHT_ONLY:
            print("GRPO skipped: GRPO_PREFLIGHT_ONLY=True.")
            iter_summary["preflight_reason"] = "GRPO_PREFLIGHT_ONLY=True; " + preflight_reason
            grpo_history.append(iter_summary)
            update_experiment_report(
                grpo_history=grpo_history,
                grpo_last_checkpoint=str(resolve_generation_checkpoint()),
                preflight_reason=iter_summary["preflight_reason"],
            )
            break

        if GRPO_EARLY_STOP_ON_COLLAPSE:
            if rollout_stats["tagged_answer_extractable_rate"] < 0.02:
                print("Early stop: tagged answer extractable rate collapsed.")
                break
            if rollout_stats["repeated_tag_rate"] > 0.95:
                print("Early stop: repeated tag rate collapsed.")
                break
            if rollout_stats["average_completion_token_length"] < 2:
                print("Early stop: completion length collapsed.")
                break

        model.train()
        losses = []
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(GRPO_EPOCHS_PER_BUFFER):
            random.Random(SEED + iteration + epoch).shuffle(rollouts)
            for start in range(0, len(rollouts), GRPO_MINIBATCH_SIZE):
                minibatch = rollouts[start:start + GRPO_MINIBATCH_SIZE]
                batch = grpo_collate(minibatch, device)
                loss = grpo_loss(model, batch) / max(1, GRPO_GRAD_ACCUM)
                loss.backward()
                losses.append(float(loss.detach().cpu()) * max(1, GRPO_GRAD_ACCUM))
                step_count += 1

                if step_count % GRPO_GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRPO_MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        if step_count % GRPO_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRPO_MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = sum(losses) / len(losses) if losses else None
        print("GRPO average loss:", avg_loss)
        save_grpo_checkpoint(model, grpo_tokenizer)

        iter_summary["avg_loss"] = avg_loss

        if GRPO_EVAL_EVERY_ITERATION and RUN_VALIDATION:
            model.eval()
            valid_outputs_iter = generate_outputs_with_loaded_model(
                model,
                grpo_tokenizer,
                valid_records,
                VALID_OUTPUT_PATH,
                max_new_tokens=MAX_NEW_TOKENS,
                retrieval_records=train_records,
            )
            iter_result = save_evaluation_report(VALID_OUTPUT_PATH, valid_records, VALID_REPORT_PATH)
            iter_summary["validation_summary"] = iter_result["summary"]
            print("GRPO iteration validation example:")
            print(json.dumps(valid_outputs_iter[0], ensure_ascii=False, indent=2)[:1200] if valid_outputs_iter else "[]")

        grpo_history.append(iter_summary)
        update_experiment_report(grpo_history=grpo_history, grpo_last_checkpoint=str(GRPO_OUTPUT_DIR))

    del model
    torch.cuda.empty_cache()
    return GRPO_OUTPUT_DIR

# %% [markdown]
# ## 7. Optional Baseline

# %%
baseline_result = None
if RUN_BASELINE_FIRST and RUN_VALIDATION:
    _ = generate_outputs(MODEL_NAME, valid_records, BASELINE_OUTPUT_PATH, MAX_NEW_TOKENS, retrieval_records=train_records)
    baseline_result = save_evaluation_report(BASELINE_OUTPUT_PATH, valid_records, BASELINE_REPORT_PATH)
    update_experiment_report(baseline_summary=baseline_result["summary"])
else:
    print("Skip baseline.")

# %% [markdown]
# ## 8. Training

# %%
if RUN_TRAIN and TRAIN_STAGE in {"sft", "sft_then_grpo", "fdd_pot"}:
    sft_start_checkpoint = resolve_sft_start_checkpoint()
    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(sft_start_checkpoint, local_files_only=True)
    )

    model = AutoModelForCausalLM.from_pretrained(sft_start_checkpoint, local_files_only=True)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_ds = SFTDataset(
        train_records,
        tokenizer,
        MAX_LENGTH,
        TRAIN_STYLE,
        PROMPT_STYLE,
        FILTER_NO_ANSWER,
        FILTER_NON_NUMERIC_GOLD,
        LOSS_STYLE,
    )
    valid_ds = SFTDataset(
        valid_records,
        tokenizer,
        MAX_LENGTH,
        TRAIN_STYLE,
        PROMPT_STYLE,
        False,
        False,
        LOSS_STYLE,
    )
    collator = PadCollator(pad_id=SAFE_EOS_ID)

    eff_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * max(1, torch.cuda.device_count())
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch) if len(train_ds) else 0
    print("train examples:", len(train_ds))
    print("valid examples:", len(valid_ds))
    print(
        f"per_device_bs={PER_DEVICE_BATCH_SIZE} | grad_accum={GRAD_ACCUM} | "
        f"gpus={torch.cuda.device_count()} | effective_batch={eff_batch} | "
        f"steps/epoch={steps_per_epoch}"
    )

    ta_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        max_steps=MAX_STEPS,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    trainer_cls = WeightedLossTrainer if LOSS_STYLE in {
        "answer_weighted",
        "light_answer_weighted",
        "full_target_or_light_weighted",
    } else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    train_dt = time.time() - t0
    print(f"\n[train] wall time: {train_dt:.1f}s ({train_dt / 60:.2f} min)")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    model_hash = sha256_dir(OUTPUT_DIR)
    (OUTPUT_DIR / "model_hash.txt").write_text(model_hash + "\n", encoding="utf-8")

    print("Saved checkpoint to:", OUTPUT_DIR)
    print("Model SHA256:", model_hash)

    reload_check = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, local_files_only=True)
    reload_check.config.pad_token_id = SAFE_EOS_ID
    reload_check.config.eos_token_id = SAFE_EOS_ID
    print("Saved model reload check OK:", type(reload_check))
    del reload_check

    train_summary = {
        "wall_time_seconds": train_dt,
        "wall_time_minutes": train_dt / 60,
        "train_examples": len(train_ds),
        "valid_examples": len(valid_ds),
        "effective_batch_size": eff_batch,
        "steps_per_epoch": steps_per_epoch,
        "sft_start_checkpoint": str(sft_start_checkpoint),
        "trainer_class": trainer_cls.__name__,
        "max_steps": MAX_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "model_hash": model_hash,
        "trainer_metrics": train_result.metrics,
    }
    update_experiment_report(training=train_summary)

    del trainer, model
    torch.cuda.empty_cache()
else:
    print(f"Skipping SFT training. RUN_TRAIN={RUN_TRAIN} TRAIN_STAGE={TRAIN_STAGE}")

# %% [markdown]
# ## 8b. Optional GRPO / RLVR Training

# %%
grpo_checkpoint_path = None
if RUN_GRPO and TRAIN_STAGE in {"grpo", "sft_then_grpo"}:
    start_ckpt = resolve_grpo_start_checkpoint()
    grpo_checkpoint_path = run_grpo_training(start_ckpt)
    print("GRPO checkpoint:", grpo_checkpoint_path)
else:
    print(f"RUN_GRPO={RUN_GRPO}; skipping GRPO training.")

# %% [markdown]
# ## 9. Validation Generation

# %%
valid_outputs = None
if RUN_VALIDATION:
    model_for_validation = resolve_generation_checkpoint()
    print("Validation model path:", model_for_validation)
    if INFERENCE_VOTING:
        valid_outputs, greedy_valid_outputs = generate_voting_outputs(
            model_for_validation,
            valid_records,
            VALID_OUTPUT_PATH,
            max_new_tokens=MAX_NEW_TOKENS,
            include_debug_metadata=True,
            greedy_output_path=GREEDY_VALID_OUTPUT_PATH,
        )
        print("Wrote greedy validation comparison file:", GREEDY_VALID_OUTPUT_PATH)
    else:
        valid_outputs = generate_outputs(
            model_for_validation,
            valid_records,
            VALID_OUTPUT_PATH,
            max_new_tokens=MAX_NEW_TOKENS,
            retrieval_records=train_records,
        )
    print("\nExample validation output:")
    print(json.dumps(valid_outputs[0], ensure_ascii=False, indent=2)[:2000] if valid_outputs else "[]")
else:
    print("RUN_VALIDATION=False; skipping validation generation.")

# %% [markdown]
# ## 10. Validation Evaluation

# %%
valid_result = None
if RUN_VALIDATION and VALID_OUTPUT_PATH.exists():
    valid_result = save_evaluation_report(VALID_OUTPUT_PATH, valid_records, VALID_REPORT_PATH)
    summary = valid_result["summary"]
    greedy_summary = None
    if INFERENCE_VOTING and GREEDY_VALID_OUTPUT_PATH.exists():
        greedy_result = evaluate(json.load(GREEDY_VALID_OUTPUT_PATH.open("r", encoding="utf-8")), valid_records)
        greedy_summary = greedy_result["summary"]
        print("\nGreedy validation comparison:")
        print(f'{greedy_summary["raw_score"]} / {greedy_summary["max_raw_score"]} ({greedy_summary["score_pct"] * 100:.2f}%)')
        print("Greedy exact count:", greedy_summary.get("exact_count"))

    update_experiment_report(
        validation_summary=summary,
        greedy_validation_summary=greedy_summary,
        inference_voting=INFERENCE_VOTING,
        num_candidates=NUM_CANDIDATES,
        run_id="original_fdd_pot_restored_inference_voting",
        base_notebook="Feedback-driven distillation.ipynb",
        retained_engineering_fixes="robust extraction, output cleanup, score_by_type, artifact checks",
        final_candidate_summary={
            "run_id": "original_fdd_pot_restored_inference_voting",
            "base_notebook": "Feedback-driven distillation.ipynb",
            "retained_engineering_fixes": "robust extraction, output cleanup, score_by_type, artifact checks",
            "train_style": TRAIN_STYLE,
            "loss_style": LOSS_STYLE,
            "inference_voting": INFERENCE_VOTING,
            "num_candidates": NUM_CANDIDATES,
            "type_routing": TYPE_ROUTING,
            "use_lora": False,
            "phase2_curriculum": False,
            "execute_pot_at_inference": FDD_EXECUTE_POT_AT_INFERENCE,
            "train_samples": len(train_records),
            "epochs": EPOCHS,
            "lr": LR,
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "runtime_train_min": load_json_if_exists(EXPERIMENT_REPORT_PATH).get("training", {}).get("wall_time_minutes"),
            "raw_score": summary.get("raw_score"),
            "score_10": summary.get("score_10"),
            "exact_count": summary.get("exact_count"),
            "exact_zero_error_count": summary.get("exact_zero_error_count"),
            "greedy_raw_score": greedy_summary.get("raw_score") if greedy_summary else None,
            "greedy_score_10": greedy_summary.get("score_10") if greedy_summary else None,
            "greedy_exact_count": greedy_summary.get("exact_count") if greedy_summary else None,
            "buckets": summary.get("buckets"),
            "score_by_type": summary.get("score_by_type"),
        },
    )
    assert VALID_OUTPUT_PATH.exists()
    assert VALID_REPORT_PATH.exists()

    print("\nFinal validation score:")
    print(f'{summary["raw_score"]} / {summary["max_raw_score"]} ({summary["score_pct"] * 100:.2f}%)')
    print(f'Score /10: {summary["score_10"]:.2f}')
    print("Exact count:", summary.get("exact_count"), "| Exact zero-error:", summary.get("exact_zero_error_count"))
    print("Extractable:", summary["extractable"], "| Numeric pairs:", summary["numeric_pairs"])
    print("Buckets:", summary["buckets"])
    print("Score by type:")
    print(json.dumps(summary["score_by_type"], ensure_ascii=False, indent=2)[:4000])
else:
    print("Validation report not created.")

# %% [markdown]
# ## 11. Error Analysis

# %%
def is_repetitive_text(text: str) -> bool:
    words = re.findall(r"\S+", text or "")
    if len(words) < 40:
        return False

    trigrams = [" ".join(words[i:i + 3]).lower() for i in range(len(words) - 2)]
    if trigrams and Counter(trigrams).most_common(1)[0][1] >= 4:
        return True

    lines = [line.strip().lower() for line in text.split("\n") if line.strip()]
    return bool(lines and Counter(lines).most_common(1)[0][1] >= 3)


def compact_example(idx: int, row: dict, pred_rec: dict, gold_rec: dict) -> dict:
    item = {
        "idx": idx,
        "id": row.get("id"),
        "type": row.get("type"),
        "score": row.get("score"),
        "rel_error": row.get("rel_error"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("pred_answer"),
        "query_vi": str(gold_rec.get("query_vi", ""))[:600],
        "gold_tail": str(gold_rec.get("response_vi", ""))[-600:],
        "model_output": str(pred_rec.get("model_output", ""))[:1000],
    }
    for key in ("candidates", "candidate_answers", "selected_answer", "selection_reason"):
        if key in pred_rec:
            item[key] = pred_rec[key]
    return item


def build_error_analysis(
    result: dict,
    pred_items: list[dict],
    gold_items: list[dict],
    limit: int = 5,
) -> dict:
    rows = result.get("rows", [])
    examples = {
        "correct_or_near_correct": [],
        "non_extractable": [],
        "extractable_but_wrong": [],
        "repetitive_outputs": [],
    }
    failure_by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "score0": 0, "non_extractable": 0})

    for i, row in enumerate(rows):
        pred_rec = pred_items[i]
        gold_rec = gold_items[i]
        typ = str(row.get("type") or "unknown")
        failure_by_type[typ]["n"] += 1
        failure_by_type[typ]["score0"] += int(row.get("score") == 0)
        failure_by_type[typ]["non_extractable"] += int(not row.get("extractable"))

        ex = compact_example(i, row, pred_rec, gold_rec)
        if row.get("score") in (10, 5) and len(examples["correct_or_near_correct"]) < limit:
            examples["correct_or_near_correct"].append(ex)
        if not row.get("extractable") and len(examples["non_extractable"]) < limit:
            examples["non_extractable"].append(ex)
        if (
            row.get("extractable")
            and row.get("score") == 0
            and row.get("rel_error") is not None
            and len(examples["extractable_but_wrong"]) < limit
        ):
            examples["extractable_but_wrong"].append(ex)
        if is_repetitive_text(pred_rec.get("model_output", "")) and len(examples["repetitive_outputs"]) < limit:
            examples["repetitive_outputs"].append(ex)

    by_type_sorted = dict(sorted(failure_by_type.items(), key=lambda kv: (-kv[1]["score0"], kv[0])))
    return {
        "summary": result.get("summary", {}),
        "examples": examples,
        "failures_by_type": by_type_sorted,
    }


error_analysis = None
if RUN_VALIDATION and valid_result is not None and VALID_OUTPUT_PATH.exists():
    with VALID_OUTPUT_PATH.open("r", encoding="utf-8") as f:
        pred_items = json.load(f)
    error_analysis = build_error_analysis(valid_result, pred_items, valid_records)
    save_json(error_analysis, ERROR_ANALYSIS_PATH)
    update_experiment_report(error_analysis_path=str(ERROR_ANALYSIS_PATH))

    print("Wrote", ERROR_ANALYSIS_PATH)
    print("Error analysis preview:")
    print(json.dumps(error_analysis["examples"], ensure_ascii=False, indent=2)[:4000])
else:
    print("Error analysis skipped.")

# %% [markdown]
# ## 12. Phase 2 Test Prediction Block

# %%
def validate_prediction_file(path: str | Path) -> None:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Prediction file must be a JSON array.")

    required = {"id", "query_vi", "type", "model_output"}
    forbidden = {
        "_target",
        "response_vi",
        "answer",
        "pred",
        "pred_answer",
        "pred_num",
        "gold",
        "gold_answer",
        "gold_num",
        "validation",
        "candidates",
        "candidate_answers",
        "selected_answer",
        "selection_reason",
        "label",
        "labels",
        "score",
        "rel_error",
    }
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction item {i} must be an object.")
        missing = required - set(item)
        if missing:
            raise ValueError(f"Prediction item {i} missing fields: {sorted(missing)}")
        leaked = forbidden & set(item)
        if leaked:
            raise ValueError(f"Prediction item {i} contains forbidden fields: {sorted(leaked)}")
        if not isinstance(item["model_output"], str):
            raise ValueError(f"Prediction item {i} model_output must be a string.")

    print(f"Prediction file format OK: {path} | n={len(items)}")


if RUN_TEST:
    if not TEST_FILE.exists():
        print("test.json not found yet; skipping Phase 2 prediction.")
    else:
        test_records = load_records(TEST_FILE)
        print("test:", len(test_records))

        test_model_path = resolve_generation_checkpoint()
        if not Path(test_model_path).exists():
            raise FileNotFoundError(
                f"{test_model_path} does not exist. For Phase 2, provide a valid trained checkpoint."
            )
        print("Test model path:", test_model_path)

        if INFERENCE_VOTING:
            test_outputs, _ = generate_voting_outputs(
                test_model_path,
                test_records,
                TEST_OUTPUT_PATH,
                max_new_tokens=MAX_NEW_TOKENS,
                include_debug_metadata=False,
            )
        else:
            test_outputs = generate_outputs(
                test_model_path,
                test_records,
                TEST_OUTPUT_PATH,
                max_new_tokens=MAX_NEW_TOKENS,
                retrieval_records=train_records,
            )
        validate_prediction_file(TEST_OUTPUT_PATH)
        update_experiment_report(
            test_prediction_path=str(TEST_OUTPUT_PATH),
            test_prediction_count=len(test_outputs),
            test_prediction_sha256=sha256_file(TEST_OUTPUT_PATH),
        )
else:
    print("RUN_TEST=False; skipping Phase 2 prediction.")

# %% [markdown]
# ## 13. Technical Report Helper

# %%
def prompt_template_for_report(prompt_style: str) -> str:
    fake = {"query_vi": "{query_vi}", "type": "{type}"}
    if uses_grpo_tagged_target(TRAIN_STYLE) or TRAIN_STAGE in {"grpo", "sft_then_grpo"}:
        return build_grpo_prompt(fake)
    return build_prompt(fake, prompt_style)


def print_technical_report_template() -> None:
    report = load_json_if_exists(EXPERIMENT_REPORT_PATH)
    val = report.get("validation_summary", {})

    validation_text = (
        f'- raw_score: {val.get("raw_score")} / {val.get("max_raw_score")}\n'
        f'- score_pct: {val.get("score_pct")}\n'
        f'- score_10: {val.get("score_10")}\n'
        f'- extractable: {val.get("extractable")}\n'
        f'- numeric_pairs: {val.get("numeric_pairs")}\n'
        f'- buckets: {val.get("buckets")}'
        if val
        else "- Validation was not run in this notebook version."
    )

    text = f"""
## Data Processing
We used only `query_vi`, `response_vi`, and optional `type` from the provided Kaggle files.
The target style for this candidate run was `{TRAIN_STYLE}`. Gold answers from training records
were used only to normalize training targets. Validation gold answers were used only for scoring
and error analysis.

## Fine-Tuning Configuration
- Base model: `NlpHUST/gpt2-vietnamese` loaded from the local Kaggle input folder
- Experiment mode: `{EXPERIMENT_MODE}`
- Train stage: `{TRAIN_STAGE}`
- Run environment: `{ACTIVE_RUN_ENV}`
- Working directory: `{WORKING_DIR}`
- Loss style: `{LOSS_STYLE}`
- Sampling style: `{SAMPLING_STYLE}`
- Train type filter: `{TRAIN_TYPE_FILTER or "all"}`
- Dedup train questions: `{DEDUP_TRAIN_QUESTIONS}`
- Retrieval context: `{USE_TRAIN_RETRIEVAL_CONTEXT}` (use only if instructor permits train-set few-shot retrieval)
- Epochs: `{EPOCHS}`
- Max length: `{MAX_LENGTH}`
- Per-device batch size: `{PER_DEVICE_BATCH_SIZE}`
- Gradient accumulation: `{GRAD_ACCUM}`
- Learning rate: `{LR}`
- Warmup ratio: `{WARMUP_RATIO}`
- Seed: `{SEED}`

## Why GRPO Was Added
Pure SFT can teach answer format and Vietnamese solution-style imitation, but validation can still show weak numeric correctness.
GRPO is included as an experimental second stage that directly optimizes verifiable final-answer correctness using train.json gold answers only.
It is a candidate approach, not guaranteed to improve over SFT.

## SFT Configuration
- SFT checkpoint dir override: `{SFT_CKPT_DIR}`
- SFT output dir: `{OUTPUT_DIR}`
- Target style: `{TRAIN_STYLE}`
- Loss style: `{LOSS_STYLE}`
- Tagged reasoning max chars/parts: `{TAGGED_REASONING_MAX_CHARS}` / `{TAGGED_REASONING_MAX_PARTS}`

## Prompt Template
Prompt style: `{PROMPT_STYLE}`

```text
{prompt_template_for_report(PROMPT_STYLE)}
```

## Target Formatting
Target style: `{TRAIN_STYLE}`.
This inference-only candidate restores the original FDD/PoT target style for reproducibility,
but does not retrain by default. It loads the existing checkpoint from `{OUTPUT_DIR}` and keeps
the model frozen.

## Inference Voting
`INFERENCE_VOTING={INFERENCE_VOTING}` generates up to `{NUM_CANDIDATES}` candidates per prompt:
greedy, low-temperature sampling, and medium-temperature sampling. The final answer is extracted
and cleaned from each candidate. If at least two normalized answers match exactly, the majority
answer is selected; otherwise the greedy answer is used. Voting metadata is saved only for
validation/debug artifacts, not in `test_predictions.json`.
- Execute generated PoT at inference: `{FDD_EXECUTE_POT_AT_INFERENCE}`
- FDD max equation steps: `{FDD_MAX_EQUATION_STEPS}`
- FDD max program lines: `{FDD_PROGRAM_MAX_LINES}`
- Keep code in output: `{FDD_KEEP_CODE_IN_OUTPUT}`

## Decoding Strategy
- Decoding style: `{DECODING_STYLE}`
- Max new tokens: `{MAX_NEW_TOKENS}`
- Repetition penalty: `{REPETITION_PENALTY}`
- No-repeat ngram size: `{NO_REPEAT_NGRAM_SIZE}`
- Postprocessing removes prompt echo, trims repeated answer lines, and keeps the first final answer line.

## GRPO Configuration
- GRPO output dir: `{GRPO_OUTPUT_DIR}`
- Run GRPO: `{RUN_GRPO}`
- Iterations: `{GRPO_ITERATIONS}`
- Rollout batch size: `{GRPO_ROLLOUT_BATCH_SIZE}`
- Num generations: `{GRPO_NUM_GENERATIONS}`
- Buffer size: `{GRPO_BUFFER_SIZE}`
- Minibatch size: `{GRPO_MINIBATCH_SIZE}`
- Epochs per buffer: `{GRPO_EPOCHS_PER_BUFFER}`
- GRPO LR: `{GRPO_LR}`
- Clip epsilon: `{GRPO_CLIP_EPS}`
- Reward weights: correctness `{GRPO_CORRECTNESS_WEIGHT}`, format `{GRPO_FORMAT_WEIGHT}`
- Sampling: temperature `{GRPO_TEMPERATURE}`, top_p `{GRPO_TOP_P}`, top_k `{GRPO_TOP_K}`
- Partial numeric reward: `{GRPO_USE_PARTIAL_NUMERIC_REWARD}`
- Format reward gated on numeric answer: `{GRPO_GATE_FORMAT_REWARD_ON_NUMERIC_ANSWER}`
- Preflight-only mode: `{GRPO_PREFLIGHT_ONLY}`
- Think/answer tags: `{GRPO_USE_THINK_ANSWER_TAGS}`

## GRPO Rollout Statistics
If GRPO was run, rollout summaries and per-iteration validation summaries are saved in:

```text
{EXPERIMENT_REPORT_PATH}
```

Preflight fields include `preflight_passed`, `preflight_reason`, tagged-answer extractability,
numeric tagged-answer rate, exact-correct rate, mean correctness reward, mean format reward,
no-signal group rate, equation-in-think rate, repeated-think rate, answer-number-in-think rate,
and malformed-tag rate. Optimizer steps are skipped when the rollout has insufficient signal.

## Validation Results
{validation_text}

## Error Analysis
We saved compact validation error analysis to:

```text
{ERROR_ANALYSIS_PATH}
```

Remaining errors should be reviewed by type, especially non-extractable answers, repetitive outputs,
and extractable numeric answers with high relative error.

## Experiments Tried
We tested candidate configurations by changing a small number of variables per Kaggle commit, such as
`TRAIN_STYLE`, `PROMPT_STYLE`, `MAX_LENGTH`, and `LR`. Validation suggested which candidate to carry
forward to a larger run.

## Final Configuration Choice
This run should be chosen only if local validation supports it. The configuration is not claimed to be
globally optimal; it is a candidate selected based on validation score, extractability, runtime, and
error analysis.

For the official Kaggle submission, keep Internet OFF, attach the required Kaggle inputs, and run the
notebook in Kaggle so the final files are produced under `/kaggle/working`.
"""
    print(text.strip())


print_technical_report_template()

print("\nOutput folder:")
working = WORKING_DIR
if working.exists():
    for p in sorted(working.glob("*")):
        try:
            print("-", p, "| size=", p.stat().st_size)
        except OSError:
            print("-", p)
else:
    print(working, "does not exist in this environment.")
