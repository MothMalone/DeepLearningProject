#!/usr/bin/env python3
"""Modular training entrypoint skeleton for the rewind3 full fine-tuning method.

The official final run remains `rewind3.ipynb` because Kaggle notebook execution
is the submission format. This script documents the same components as a normal
research-code entrypoint and supports small smoke runs when the Kaggle inputs are
available locally.
"""

from __future__ import annotations

import argparse
import inspect
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.vn_gpt2_math.config import SAFE_EOS_ID, TrainingConfig
from src.vn_gpt2_math.data import load_records, sha256_dir
from src.vn_gpt2_math.dataset import PadCollator, SFTDataset
from src.vn_gpt2_math.targets import attach_targets
from src.vn_gpt2_math.training import WeightedLossTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(epochs=args.epochs)

    train_records = load_records(args.train)
    valid_records = load_records(args.valid)
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    if args.max_valid_samples:
        valid_records = valid_records[: args.max_valid_samples]
    train_records = attach_targets(train_records)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.pad_token_id = SAFE_EOS_ID
    tokenizer.eos_token_id = SAFE_EOS_ID
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.config.use_cache = False

    train_ds = SFTDataset(train_records, tokenizer, cfg)
    valid_ds = SFTDataset(valid_records, tokenizer, cfg)
    collator = PadCollator()
    effective_batch = cfg.per_device_batch_size * cfg.grad_accum * max(1, torch.cuda.device_count())
    print(f"train={len(train_ds)} valid={len(valid_ds)} effective_batch={effective_batch}")
    print(f"steps_per_epoch={math.ceil(len(train_ds) / effective_batch)}")

    ta_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size * 2,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=cfg.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=cfg.epochs,
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    ta_kwargs["eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"] = "epoch"

    trainer = WeightedLossTrainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    if args.dry_run:
        print("dry-run requested; constructed trainer but did not start training")
        return

    start = time.time()
    trainer.train()
    print(f"train_minutes={(time.time() - start) / 60:.2f}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("model_hash", sha256_dir(args.output_dir))


if __name__ == "__main__":
    main()
