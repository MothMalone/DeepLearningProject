#!/usr/bin/env python3
"""Audit Vietnamese math dataset quality before training.

This script is intentionally lightweight: it reads train/valid JSON files,
counts high-risk formatting patterns, and writes a compact JSON report. It does
not modify raw data and does not import training code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vn_gpt2_math.answers import VI_ANCHORS, extract_answer, parse_number
from src.vn_gpt2_math.targets import clean_math_text

LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
COMPOUND_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{[^{}]*[+\-*/=][^{}]*\}\s*\{[^{}]*[+\-*/=][^{}]*\}")
ASY_RE = re.compile(r"\[asy\].*?(?:\[/asy\]|$)", re.IGNORECASE | re.DOTALL)
INEQUALITY_RE = re.compile(r"\\(?:leq?|geq?|neq?|equiv)\b")
MATH_FUNCTION_RE = re.compile(r"\\(?:cos|sin|tan|log|ln|sqrt|pi|theta|alpha|beta|gamma|delta)\b")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like records in {path}")
    return [r for r in data if isinstance(r, dict)]


def legacy_latex_cleanup(text: str) -> str:
    """Approximate the old notebook cleanup for before/after risk examples."""
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1/\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\times", " * ").replace("\\cdot", " * ")
    text = re.sub(r"\\angle\s*([A-Z]{2,3})", r"góc \1", text)
    text = re.sub(r"\\triangle\s*([A-Z]{2,3})", r"tam giác \1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\s", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def audit_split(name: str, records: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    latex_commands: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        query = rec.get("query_vi", "")
        response = rec.get("response_vi", "")
        typ = str(rec.get("type") or "UNKNOWN")
        type_counts[typ] += 1

        if not query:
            counts["missing_query"] += 1
        if not response:
            counts["missing_response"] += 1

        gold = extract_answer(response, VI_ANCHORS)
        gold_num = parse_number(gold)
        counts["gold_extractable"] += int(gold is not None)
        counts["gold_numeric"] += int(gold_num is not None)

        joined = f"{query}\n{response}"
        commands = LATEX_CMD_RE.findall(joined)
        latex_commands.update(commands)
        counts["records_with_latex"] += int(bool(commands))
        counts["frac_total"] += len(FRAC_RE.findall(joined))
        counts["compound_frac_records"] += int(bool(COMPOUND_FRAC_RE.search(joined)))
        counts["asy_records"] += int(bool(ASY_RE.search(joined)))
        counts["inequality_records"] += int(bool(INEQUALITY_RE.search(joined)))
        counts["math_function_records"] += int(bool(MATH_FUNCTION_RE.search(joined)))

        risky = bool(COMPOUND_FRAC_RE.search(joined) or ASY_RE.search(joined) or INEQUALITY_RE.search(joined))
        if risky and len(examples) < max_examples:
            fixed = clean_math_text(joined[:2500])
            legacy = legacy_latex_cleanup(joined[:2500])
            examples.append(
                {
                    "idx": idx,
                    "type": typ,
                    "risk": {
                        "compound_fraction": bool(COMPOUND_FRAC_RE.search(joined)),
                        "asy_block": bool(ASY_RE.search(joined)),
                        "inequality_command": bool(INEQUALITY_RE.search(joined)),
                    },
                    "raw_excerpt": joined[:700],
                    "legacy_excerpt": legacy[:700],
                    "fixed_excerpt": fixed[:700],
                }
            )

    return {
        "split": name,
        "records": len(records),
        "type_counts": dict(type_counts),
        "counts": dict(counts),
        "rates": {
            "gold_numeric_rate": counts["gold_numeric"] / max(1, len(records)),
            "records_with_latex_rate": counts["records_with_latex"] / max(1, len(records)),
            "compound_frac_record_rate": counts["compound_frac_records"] / max(1, len(records)),
            "asy_record_rate": counts["asy_records"] / max(1, len(records)),
            "inequality_record_rate": counts["inequality_records"] / max(1, len(records)),
        },
        "top_latex_commands": dict(latex_commands.most_common(30)),
        "risk_examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train.json and valid.json")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--max-examples", type=int, default=12, help="Number of risk examples per split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.json"
    valid_path = args.data_dir / "valid.json"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Expected train.json and valid.json under {args.data_dir}")

    report = {
        "data_dir": str(args.data_dir),
        "train": audit_split("train", load_records(train_path), args.max_examples),
        "valid": audit_split("valid", load_records(valid_path), args.max_examples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    train = report["train"]
    valid = report["valid"]
    print(f"Wrote {args.output}")
    print(
        "train:",
        {
            "records": train["records"],
            "gold_numeric": train["counts"].get("gold_numeric", 0),
            "records_with_latex": train["counts"].get("records_with_latex", 0),
            "compound_frac_records": train["counts"].get("compound_frac_records", 0),
            "asy_records": train["counts"].get("asy_records", 0),
            "inequality_records": train["counts"].get("inequality_records", 0),
        },
    )
    print(
        "valid:",
        {
            "records": valid["records"],
            "gold_numeric": valid["counts"].get("gold_numeric", 0),
            "records_with_latex": valid["counts"].get("records_with_latex", 0),
            "compound_frac_records": valid["counts"].get("compound_frac_records", 0),
            "asy_records": valid["counts"].get("asy_records", 0),
            "inequality_records": valid["counts"].get("inequality_records", 0),
        },
    )


if __name__ == "__main__":
    main()
