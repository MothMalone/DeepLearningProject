#!/usr/bin/env python3
"""Select a compact high-quality subset from preprocessed JSONL records."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from preprocess_utils import contains_operator, parse_number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--strategy",
        choices=["all_usable", "random_usable", "quality_top", "quality_diverse"],
        default="quality_diverse",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def quality_score(rec: dict) -> int:
    target = str(rec.get("target_direct", ""))
    lines = rec.get("solution_lines", []) or []
    score = 0
    score += int(bool(lines)) * 2
    score += int(contains_operator(lines)) * 2
    score += int(len(target) <= 350)
    score += int(not rec.get("fallback_used"))
    score += int(parse_number(str(rec.get("gold_answer", ""))) is not None)
    score += int(str(rec.get("gold_answer", "")) in target)
    return score


def magnitude_bucket(rec: dict) -> str:
    val = parse_number(str(rec.get("gold_answer", "")))
    if val is None:
        return "non_numeric"
    aval = abs(val)
    if aval < 1:
        return "<1"
    if aval < 10:
        return "1-9"
    if aval < 100:
        return "10-99"
    if aval < 1000:
        return "100-999"
    return "1000+"


def question_len_bucket(rec: dict) -> str:
    n = len(str(rec.get("query_vi_original", rec.get("query_vi_clean", ""))))
    if n < 180:
        return "short"
    if n < 450:
        return "medium"
    return "long"


def operation_signature(rec: dict) -> str:
    text = "\n".join(rec.get("solution_lines", []) or []) + "\n" + str(rec.get("query_vi_original", ""))
    sig = []
    if re.search(r"\+|thêm|cộng", text, re.IGNORECASE):
        sig.append("add")
    if re.search(r"\-|bớt|trừ|mất|giảm", text, re.IGNORECASE):
        sig.append("sub")
    if re.search(r"\*|×|nhân|gấp", text, re.IGNORECASE):
        sig.append("mul")
    if re.search(r"/|chia|mỗi", text, re.IGNORECASE):
        sig.append("div")
    if re.search(r"%|phần trăm", text, re.IGNORECASE):
        sig.append("percent")
    if re.search(r"\\frac|\d+\s*/\s*\d+", text):
        sig.append("fraction")
    if "=" in text:
        sig.append("equation")
    return "+".join(sig) if sig else "other"


def diversity_key(rec: dict) -> tuple[str, str, str, str]:
    return (
        str(rec.get("type", "missing")),
        magnitude_bucket(rec),
        question_len_bucket(rec),
        operation_signature(rec),
    )


def quality_diverse(records: list[dict], max_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for rec in sorted(records, key=lambda r: quality_score(r), reverse=True):
        buckets[diversity_key(rec)].append(rec)

    for bucket in buckets.values():
        rng.shuffle(bucket)
        bucket.sort(key=lambda r: quality_score(r), reverse=True)

    selected: list[dict] = []
    keys = list(buckets)
    rng.shuffle(keys)
    while len(selected) < max_examples and keys:
        next_keys: list[tuple[str, str, str, str]] = []
        for key in keys:
            if buckets[key] and len(selected) < max_examples:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def main() -> int:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    records = [r for r in read_jsonl(args.input) if r.get("usable") and r.get("target_direct")]
    limit = args.max_examples or len(records)

    if args.strategy == "all_usable":
        selected = records[:limit]
    elif args.strategy == "random_usable":
        selected = list(records)
        rng.shuffle(selected)
        selected = selected[:limit]
    elif args.strategy == "quality_top":
        selected = sorted(records, key=lambda r: quality_score(r), reverse=True)[:limit]
    else:
        selected = quality_diverse(records, limit, args.seed)

    write_jsonl(args.output, selected)
    print("Input:", args.input)
    print("Usable input records:", len(records))
    print("Strategy:", args.strategy)
    print("Selected:", len(selected))
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
