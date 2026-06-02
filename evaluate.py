#!/usr/bin/env python3
"""Evaluate a prediction JSON file against valid.json using the competition metric."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.vn_gpt2_math.data import load_records, save_json
from src.vn_gpt2_math.metrics import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("valid_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_records(args.predictions)
    gold = load_records(args.gold)
    report = evaluate(preds, gold)
    save_json(report, args.output)
    summary = report["summary"]
    print(f"raw_score={summary['raw_score']}/{summary['max_raw_score']}")
    print(f"exact_count={summary['exact_count']} extractable={summary['extractable']}/{summary['n']}")


if __name__ == "__main__":
    main()
