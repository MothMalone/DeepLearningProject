#!/usr/bin/env python3
"""Distill train.json into cleaner GPT-2 SFT targets with a local HF model.

This script is intentionally separate from the Kaggle notebook. It is meant for
Vast.ai/local preprocessing experiments and writes an inspectable JSONL dataset
that the notebook can optionally consume through TRAIN_DATA_OVERRIDE.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess_utils import (
    cleanup_answer_string,
    contains_operator,
    extract_answer,
    extract_json_object,
    load_records,
    normalize_space,
    numbers_match,
    parse_number,
    sha256_text,
)


PROMPT_VERSION = "v1_compact_json_cleaner"
MAX_TARGET_CHARS = 500
META_TEXT_RE = re.compile(
    r"\bAI\b|tôi không thể|không đủ thông tin|as an assistant|language model",
    flags=re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to train.json or train JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--report", required=True, help="Output preprocessing report JSON path.")
    parser.add_argument("--model", required=True, help="Local model path or HuggingFace model name.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--type-filter", choices=["", "GSM", "MATH"], default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-chars", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--fallback-answer-only", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--strict-json", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--save-raw-generations", action="store_true")
    parser.add_argument("--allow-non-scalar", action="store_true")
    parser.add_argument("--report-every", type=int, default=50)
    return parser


def dtype_from_arg(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def model_source_hash(model_name: str) -> str:
    p = Path(model_name)
    if not p.exists():
        return sha256_text(model_name)
    parts: list[str] = []
    for child in sorted(x for x in p.rglob("*") if x.is_file() and x.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}):
        try:
            parts.append(f"{child.relative_to(p)}:{child.stat().st_size}:{int(child.stat().st_mtime)}")
        except OSError:
            continue
    return sha256_text("\n".join(parts) or str(p.resolve()))


def load_teacher_model(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Refusing CPU preprocessing unless --allow-cpu is passed.")

    quantization_config = None
    load_kwargs: dict = {"torch_dtype": dtype_from_arg(args.torch_dtype)}
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("4-bit/8-bit loading requires bitsandbytes support in transformers.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs.pop("torch_dtype", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        **load_kwargs,
    )
    if device == "cuda" and quantization_config is None:
        model = model.to(device)
    model.eval()
    return tokenizer, model


def build_cleaning_prompt(query: str, original_solution: str, gold_answer: str, max_input_chars: int) -> str:
    query = normalize_space(query)[:max_input_chars]
    original_solution = normalize_space(original_solution)[:max_input_chars]
    return f"""You are cleaning a Vietnamese math training example for a weak GPT-2 model.

Return only valid JSON.

Goal:
Create a short supervised fine-tuning target with compact calculation lines.

Rules:
- Use only the given question, original solution, and verified final answer.
- Do not solve a different problem.
- The final_answer field must exactly equal the verified gold answer.
- solution_lines should be short equations or arithmetic steps.
- Maximum 4 solution lines.
- Each solution line should be short.
- Avoid long prose.
- Avoid XML/HTML/Markdown.
- Avoid tags like <think>, <answer>, <td>, <br>.
- Do not mention that you are an AI or that this is preprocessing.
- If the original solution is too messy, use an empty solution_lines list.
- Do not include any text outside JSON.

JSON schema:
{{
  "clean_question": "...",
  "solution_lines": ["...", "..."],
  "final_answer": "...",
  "usable": true
}}

Question:
{query}

Original solution:
{original_solution}

Verified gold answer:
{gold_answer}
"""


def apply_chat_template_if_available(tokenizer, prompts: list[str]) -> list[str]:
    if getattr(tokenizer, "chat_template", None):
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    return prompts


@dataclass
class PreparedExample:
    source_index: int
    record: dict
    query: str
    response: str
    gold_answer: str
    gold_num: Optional[float]
    source_hash: str


def stable_source_hash(query: str, response: str, gold_answer: str, model_name: str) -> str:
    return sha256_text("\n".join([query, response, gold_answer, model_name, PROMPT_VERSION]))


def prepare_examples(records: list[dict], args: argparse.Namespace) -> tuple[list[PreparedExample], Counter]:
    counts: Counter = Counter()
    prepared: list[PreparedExample] = []
    for idx, rec in enumerate(records):
        counts["total_seen"] += 1
        if idx < args.start_index:
            continue
        typ = str(rec.get("type", "") or "")
        if args.type_filter and not typ.startswith(args.type_filter):
            continue

        query = normalize_space(rec.get("query_vi"))
        response = normalize_space(rec.get("response_vi"))
        if not query or not response:
            counts["skipped_missing_query_or_response"] += 1
            continue

        gold_answer = extract_answer(response)
        if not gold_answer:
            counts["skipped_missing_answer"] += 1
            continue
        gold_answer = cleanup_answer_string(gold_answer)
        gold_num = parse_number(gold_answer)
        if gold_num is None and not args.allow_non_scalar:
            counts["skipped_non_numeric"] += 1
            continue

        prepared.append(
            PreparedExample(
                source_index=idx,
                record=rec,
                query=query,
                response=response,
                gold_answer=gold_answer,
                gold_num=gold_num,
                source_hash=stable_source_hash(query, response, gold_answer, args.model),
            )
        )
        if args.max_examples and len(prepared) >= args.max_examples:
            break
    return prepared, counts


def read_existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = obj.get("source_hash")
            if isinstance(h, str):
                hashes.add(h)
    return hashes


def make_target(solution_lines: list[str], gold_answer: str) -> tuple[str, str]:
    answer_only = f"Đáp án là: {gold_answer}"
    if not solution_lines:
        return answer_only, answer_only
    body = "\n".join(solution_lines)
    return f"Lời giải ngắn:\n{body}\nĐáp án là: {gold_answer}", answer_only


def normalize_solution_lines(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    lines: list[str] = []
    for item in value[:4]:
        line = normalize_space(str(item))
        if line:
            lines.append(line[:160])
    return lines


def validate_cleaned(obj: dict, example: PreparedExample) -> tuple[bool, Optional[str], dict]:
    final_answer = cleanup_answer_string(str(obj.get("final_answer", "")))
    if not final_answer:
        return False, "missing_final_answer", {}
    if not numbers_match(final_answer, example.gold_answer):
        return False, "final_answer_mismatch", {"final_answer": final_answer}

    clean_question = normalize_space(str(obj.get("clean_question", "")))
    if not clean_question:
        return False, "empty_clean_question", {}

    solution_lines = normalize_solution_lines(obj.get("solution_lines"))
    if len(solution_lines) > 4:
        return False, "too_many_solution_lines", {}

    target_direct, target_answer_only = make_target(solution_lines, example.gold_answer)
    combined = "\n".join([clean_question, *solution_lines, target_direct])
    if len(target_direct) > MAX_TARGET_CHARS:
        return False, "target_too_long", {"target_chars": len(target_direct)}
    if TAG_RE.search(combined):
        return False, "html_or_xml_tag_detected", {}
    if "```" in combined:
        return False, "markdown_fence_detected", {}
    if META_TEXT_RE.search(combined):
        return False, "meta_text_detected", {}
    if solution_lines and not any(re.search(r"\d", x) or contains_operator(x) for x in solution_lines):
        return False, "solution_lines_without_numeric_signal", {}

    return True, None, {
        "clean_question": clean_question,
        "solution_lines": solution_lines,
        "final_answer": example.gold_answer,
        "target_direct": target_direct,
        "target_answer_only": target_answer_only,
    }


def fallback_payload(example: PreparedExample, source_model: str, source_model_hash: str, error: str, raw_generation: str | None) -> dict:
    target_direct, target_answer_only = make_target([], example.gold_answer)
    payload = {
        "id": example.record.get("id", example.source_index),
        "type": example.record.get("type"),
        "query_vi_original": example.query,
        "query_vi_clean": example.query,
        "response_vi_original": example.response,
        "gold_answer": example.gold_answer,
        "gold_num": example.gold_num,
        "solution_lines": [],
        "target_direct": target_direct,
        "target_answer_only": target_answer_only,
        "usable": True,
        "fallback_used": True,
        "validation_error": error,
        "source_model": source_model,
        "source_hash": example.source_hash,
        "source_model_hash": source_model_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw_generation is not None:
        payload["raw_generation"] = raw_generation
    return payload


def unusable_payload(example: PreparedExample, source_model: str, source_model_hash: str, error: str, raw_generation: str | None) -> dict:
    payload = fallback_payload(example, source_model, source_model_hash, error, raw_generation)
    payload["usable"] = False
    payload["fallback_used"] = False
    payload["target_direct"] = ""
    payload["target_answer_only"] = f"Đáp án là: {example.gold_answer}"
    return payload


def cleaned_payload(example: PreparedExample, cleaned: dict, source_model: str, source_model_hash: str, raw_generation: str | None) -> dict:
    payload = {
        "id": example.record.get("id", example.source_index),
        "type": example.record.get("type"),
        "query_vi_original": example.query,
        "query_vi_clean": cleaned["clean_question"],
        "response_vi_original": example.response,
        "gold_answer": example.gold_answer,
        "gold_num": example.gold_num,
        "solution_lines": cleaned["solution_lines"],
        "target_direct": cleaned["target_direct"],
        "target_answer_only": cleaned["target_answer_only"],
        "usable": True,
        "fallback_used": False,
        "validation_error": None,
        "source_model": source_model,
        "source_hash": example.source_hash,
        "source_model_hash": source_model_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw_generation is not None:
        payload["raw_generation"] = raw_generation
    return payload


def generate_batch(tokenizer, model, prompts: list[str], args: argparse.Namespace) -> list[str]:
    model_prompts = apply_chat_template_if_available(tokenizer, prompts)
    device = next(model.parameters()).device
    enc = tokenizer(model_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    do_sample = args.temperature > 0
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        output_ids = model.generate(**enc, **gen_kwargs)
    texts: list[str] = []
    prompt_width = enc["input_ids"].shape[1]
    for i in range(output_ids.shape[0]):
        new_ids = output_ids[i, prompt_width:]
        texts.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return texts


def summarize_report(records: list[dict], counts: Counter, args: argparse.Namespace, output_path: Path, report_path: Path) -> dict:
    usable_records = [r for r in records if r.get("usable")]
    fallback_records = [r for r in records if r.get("fallback_used")]
    target_lengths = [len(str(r.get("target_direct", ""))) for r in usable_records if r.get("target_direct")]
    equation_records = [r for r in usable_records if contains_operator(r.get("solution_lines", []))]
    type_counts = Counter(str(r.get("type", "missing")) for r in records)

    def pct(num: int, den: int) -> float:
        return float(num) / den if den else 0.0

    report = {
        "input": args.input,
        "output": str(output_path),
        "report": str(report_path),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "max_examples": args.max_examples,
        "type_filter": args.type_filter,
        "counts": dict(counts),
        "rates": {
            "usable_rate": pct(len(usable_records), len(records)),
            "fallback_rate": pct(len(fallback_records), len(records)),
            "json_parse_fail_rate": pct(counts.get("json_parse_fail", 0), max(1, counts.get("llm_attempted", 0))),
            "validation_fail_rate": pct(counts.get("validation_fail", 0), max(1, counts.get("llm_attempted", 0))),
            "equation_line_rate": pct(len(equation_records), len(usable_records)),
        },
        "target_stats": {
            "avg_target_chars": statistics.mean(target_lengths) if target_lengths else 0.0,
            "p50_target_chars": statistics.median(target_lengths) if target_lengths else 0.0,
            "p95_target_chars": sorted(target_lengths)[int(0.95 * (len(target_lengths) - 1))] if target_lengths else 0.0,
        },
        "type_counts": dict(type_counts.most_common()),
        "sample_outputs": usable_records[:5],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()
    if args.overwrite and report_path.exists():
        report_path.unlink()

    records = load_records(args.input)
    prepared, counts = prepare_examples(records, args)
    existing_hashes = read_existing_hashes(output_path) if args.resume else set()
    counts["prepared_examples"] = len(prepared)
    counts["skipped_existing"] = sum(1 for ex in prepared if ex.source_hash in existing_hashes)

    source_model_hash = model_source_hash(args.model)
    tokenizer, model = load_teacher_model(args)

    all_output_records: list[dict] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            all_output_records = [json.loads(line) for line in f if line.strip()]

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as out_f:
        pending = [ex for ex in prepared if ex.source_hash not in existing_hashes]
        for start in tqdm(range(0, len(pending), args.batch_size), desc="Preprocessing"):
            batch = pending[start : start + args.batch_size]
            prompts = [
                build_cleaning_prompt(ex.query, ex.response, ex.gold_answer, args.max_input_chars)
                for ex in batch
            ]
            counts["llm_attempted"] += len(batch)
            raw_outputs = generate_batch(tokenizer, model, prompts, args)
            for ex, raw in zip(batch, raw_outputs):
                raw_keep = raw if args.save_raw_generations else None
                obj = extract_json_object(raw)
                if obj is None:
                    counts["json_parse_fail"] += 1
                    if args.fallback_answer_only:
                        payload = fallback_payload(ex, args.model, source_model_hash, "json_parse_fail", raw_keep)
                        counts["fallback_used"] += 1
                        counts["usable"] += 1
                    else:
                        payload = unusable_payload(ex, args.model, source_model_hash, "json_parse_fail", raw_keep)
                        counts["unusable"] += 1
                    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    all_output_records.append(payload)
                    continue

                ok, error, cleaned = validate_cleaned(obj, ex)
                if not ok:
                    counts["validation_fail"] += 1
                    if args.fallback_answer_only:
                        payload = fallback_payload(ex, args.model, source_model_hash, error or "validation_fail", raw_keep)
                        counts["fallback_used"] += 1
                        counts["usable"] += 1
                    else:
                        payload = unusable_payload(ex, args.model, source_model_hash, error or "validation_fail", raw_keep)
                        counts["unusable"] += 1
                    out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    all_output_records.append(payload)
                    continue

                payload = cleaned_payload(ex, cleaned, args.model, source_model_hash, raw_keep)
                counts["llm_success"] += 1
                counts["usable"] += 1
                out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                all_output_records.append(payload)

            out_f.flush()
            if args.report_every > 0 and (start + len(batch)) % args.report_every == 0:
                summarize_report(all_output_records, counts, args, output_path, report_path)

    report = summarize_report(all_output_records, counts, args, output_path, report_path)
    print("\nPreprocessing summary")
    print(json.dumps({k: report[k] for k in ["model", "prompt_version", "counts", "rates", "target_stats"]}, ensure_ascii=False, indent=2))
    print("Output:", output_path)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
