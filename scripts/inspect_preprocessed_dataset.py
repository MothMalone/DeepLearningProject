#!/usr/bin/env python3
"""Inspect JSONL produced by preprocess_with_local_llm.py."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from preprocess_utils import contains_operator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--only-fallback", action="store_true")
    parser.add_argument("--only-usable", action="store_true")
    parser.add_argument("--type", default="")
    parser.add_argument("--contains-operator", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    args = build_parser().parse_args()
    random.seed(args.seed)
    records = read_jsonl(args.input)

    usable = [r for r in records if r.get("usable")]
    fallback = [r for r in records if r.get("fallback_used")]
    target_lengths = [len(str(r.get("target_direct", ""))) for r in usable if r.get("target_direct")]

    filtered = list(records)
    if args.only_fallback:
        filtered = [r for r in filtered if r.get("fallback_used")]
    if args.only_usable:
        filtered = [r for r in filtered if r.get("usable")]
    if args.type:
        filtered = [r for r in filtered if str(r.get("type", "")) == args.type]
    if args.contains_operator:
        filtered = [r for r in filtered if contains_operator(r.get("solution_lines", []))]

    print("Input:", args.input)
    print("Total records:", len(records))
    print("Usable:", len(usable))
    print("Fallback:", len(fallback))
    print("Average target length:", round(sum(target_lengths) / len(target_lengths), 2) if target_lengths else 0.0)
    print("Type distribution:", dict(Counter(str(r.get("type", "missing")) for r in records).most_common()))
    print("Filtered records:", len(filtered))

    sample_n = min(args.n, len(filtered))
    samples = random.sample(filtered, sample_n) if sample_n else []
    for i, rec in enumerate(samples, 1):
        print("\n" + "=" * 100)
        print(f"Example {i} | id={rec.get('id')} | type={rec.get('type')} | fallback={rec.get('fallback_used')}")
        print("Question:", str(rec.get("query_vi_original", rec.get("query_vi_clean", "")))[:700])
        print("Gold answer:", rec.get("gold_answer"), "| gold_num:", rec.get("gold_num"))
        print("Solution lines:")
        for line in rec.get("solution_lines", []) or []:
            print(" -", line)
        print("Target:")
        print(str(rec.get("target_direct", ""))[:1000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
