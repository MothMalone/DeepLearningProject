#!/usr/bin/env python3
"""Frozen checkpoint inference for Vietnamese GPT-2 math predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.vn_gpt2_math.config import SAFE_EOS_ID, DecodingConfig
from src.vn_gpt2_math.data import load_records, save_json
from src.vn_gpt2_math.generation import generate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.pad_token_id = SAFE_EOS_ID
    tokenizer.eos_token_id = SAFE_EOS_ID
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True)
    model.config.pad_token_id = SAFE_EOS_ID
    model.config.eos_token_id = SAFE_EOS_ID
    model.to(device)
    model.eval()

    config = DecodingConfig(
        max_new_tokens=args.max_new_tokens,
        use_beam_search=False,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    records = load_records(args.input)
    outputs = generate_outputs(model, tokenizer, records, config, batch_size=args.batch_size, device=device)
    safe_outputs = [
        {"id": item.get("id", i), "query_vi": item["query_vi"], "type": item.get("type"), "model_output": item["model_output"]}
        for i, item in enumerate(outputs)
    ]
    save_json(safe_outputs, args.output)


if __name__ == "__main__":
    main()
