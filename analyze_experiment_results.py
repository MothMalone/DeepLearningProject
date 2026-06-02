#!/usr/bin/env python3
"""Aggregate Vast.ai experiment outputs.

Reads experiment directories containing valid_report.json / valid_output.json and
writes compact JSON + Markdown summaries. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_repetitive_text(text: str) -> bool:
    words = re.findall(r"\S+", text or "")
    if len(words) < 40:
        return False
    trigrams = [" ".join(words[i:i + 3]).lower() for i in range(len(words) - 2)]
    if trigrams and Counter(trigrams).most_common(1)[0][1] >= 4:
        return True
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    return bool(lines and Counter(lines).most_common(1)[0][1] >= 3)


def compact_example(row: dict[str, Any], pred: Optional[dict[str, Any]]) -> dict[str, Any]:
    pred = pred or {}
    return {
        "id": row.get("id"),
        "type": row.get("type"),
        "score": row.get("score"),
        "rel_error": row.get("rel_error"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("pred_answer"),
        "query_vi": str(pred.get("query_vi", ""))[:500],
        "model_output": str(pred.get("model_output", ""))[:900],
    }


def summarize_run(run_dir: Path, max_examples: int = 5) -> Optional[dict[str, Any]]:
    report_path = run_dir / "valid_report.json"
    if not report_path.exists():
        return None

    report = read_json(report_path)
    rows = report.get("rows", [])
    summary = dict(report.get("summary", {}))

    preds = []
    pred_path = run_dir / "valid_output.json"
    if pred_path.exists():
        preds = read_json(pred_path)

    bad_extract = []
    bad_parse = []
    bad_wrong = []
    good = []
    partial = []
    repetitive = []
    quoted_answer = []
    failures_by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "score0": 0, "non_extractable": 0, "bad_parse": 0}
    )

    for row in rows:
        idx = row.get("id")
        pred = preds[int(idx)] if idx is not None and preds and int(idx) < len(preds) else None
        output = str((pred or {}).get("model_output", ""))
        typ = str(row.get("type") or "unknown")
        failures_by_type[typ]["n"] += 1

        if row.get("score") == 10:
            good.append((row, pred))
        elif row.get("score") in (1, 5):
            partial.append((row, pred))

        if not row.get("extractable"):
            bad_extract.append((row, pred))
            failures_by_type[typ]["non_extractable"] += 1
        elif row.get("gold_num") is None or row.get("pred_num") is None:
            bad_parse.append((row, pred))
            failures_by_type[typ]["bad_parse"] += 1
        elif row.get("score") == 0:
            bad_wrong.append((row, pred))

        if row.get("score") == 0:
            failures_by_type[typ]["score0"] += 1
        if is_repetitive_text(output):
            repetitive.append((row, pred))
        if re.search(r"Đáp\s*án\s*là\s*:\s*[\"'“”‘’`]", output):
            quoted_answer.append((row, pred))

    examples = {
        "good": [compact_example(r, p) for r, p in good[:max_examples]],
        "partial": [compact_example(r, p) for r, p in partial[:max_examples]],
        "non_extractable": [compact_example(r, p) for r, p in bad_extract[:max_examples]],
        "extractable_not_numeric": [compact_example(r, p) for r, p in bad_parse[:max_examples]],
        "extractable_wrong": [compact_example(r, p) for r, p in bad_wrong[:max_examples]],
        "repetitive": [compact_example(r, p) for r, p in repetitive[:max_examples]],
        "quoted_answer": [compact_example(r, p) for r, p in quoted_answer[:max_examples]],
    }

    experiment_report = {}
    exp_path = run_dir / "experiment_report.json"
    if exp_path.exists():
        experiment_report = read_json(exp_path)

    config = dict(experiment_report.get("config", {}))
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = read_json(run_config_path)
        for key, value in run_config.items():
            config.setdefault(key, value)

    status = {}
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        status = read_json(status_path)

    n = int(summary.get("n") or 0)
    buckets = summary.get("buckets", {}) or {}
    bucket_10 = int(buckets.get("10", 0) or 0)
    bucket_5 = int(buckets.get("5", 0) or 0)
    bucket_1 = int(buckets.get("1", 0) or 0)
    bucket_0 = int(buckets.get("0", 0) or 0)
    extractable = int(summary.get("extractable") or 0)
    numeric_pairs = int(summary.get("numeric_pairs") or 0)
    training = experiment_report.get("training", {})

    metrics = {
        "n": n,
        "score_10": float(summary.get("score_10") or 0.0),
        "score_pct": float(summary.get("score_pct") or 0.0),
        "raw_score": int(summary.get("raw_score") or summary.get("total_score") or 0),
        "max_raw_score": int(summary.get("max_raw_score") or summary.get("max_score") or 0),
        "bucket_10_count": bucket_10,
        "bucket_5_count": bucket_5,
        "bucket_1_count": bucket_1,
        "bucket_0_count": bucket_0,
        "partial_count": bucket_5 + bucket_1,
        "bucket_10_rate": bucket_10 / n if n else 0.0,
        "extractable_rate": extractable / n if n else 0.0,
        "numeric_pair_rate": numeric_pairs / n if n else 0.0,
        "runtime_minutes": training.get("wall_time_minutes"),
        "preflight_passed": experiment_report.get("preflight_passed"),
        "preflight_reason": experiment_report.get("preflight_reason"),
        "tagged_answer_extractable_rate": experiment_report.get("tagged_answer_extractable_rate"),
        "numeric_tagged_answer_rate": experiment_report.get("numeric_tagged_answer_rate"),
        "exact_correct_rate": experiment_report.get("exact_correct_rate"),
        "mean_correctness_reward": experiment_report.get("mean_correctness_reward"),
        "mean_format_reward": experiment_report.get("mean_format_reward"),
        "no_signal_group_rate": experiment_report.get("no_signal_group_rate"),
        "has_equation_in_think_rate": experiment_report.get("has_equation_in_think_rate"),
        "repeated_think_rate": experiment_report.get("repeated_think_rate"),
        "answer_number_in_think_rate": experiment_report.get("answer_number_in_think_rate"),
        "malformed_tag_rate": experiment_report.get("malformed_tag_rate"),
    }

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "config": config,
        "summary": summary,
        "metrics": metrics,
        "diagnostics": {
            "good_count": len(good),
            "partial_count": len(partial),
            "non_extractable_count": len(bad_extract),
            "extractable_not_numeric_count": len(bad_parse),
            "extractable_wrong_count": len(bad_wrong),
            "repetitive_count": len(repetitive),
            "quoted_answer_count": len(quoted_answer),
            "failures_by_type": dict(sorted(failures_by_type.items())),
        },
        "examples": examples,
    }


def score_key(run: dict[str, Any]) -> tuple[float, float, float, float]:
    m = run.get("metrics", {})
    return (
        float(m.get("score_10") or 0.0),
        float(m.get("bucket_10_rate") or 0.0),
        float(m.get("extractable_rate") or 0.0),
        float(m.get("numeric_pair_rate") or 0.0),
    )


def make_markdown(runs: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment Summary",
        "",
        "| Rank | Run | n | score_10 | raw_score | 10 | partial | 0 | extractable_rate | numeric_pair_rate | stage | train_style | prompt_style | loss_style | sampling | dedup | decoding | max_train | max_len | new_tok | lr | runtime_min |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    ranked = sorted(runs, key=score_key, reverse=True)
    for rank, run in enumerate(ranked, 1):
        m = run.get("metrics", {})
        cfg = run.get("config", {})
        raw = f"{m.get('raw_score')}/{m.get('max_raw_score')}"
        runtime = m.get("runtime_minutes")
        runtime_text = "" if runtime is None else f"{float(runtime):.2f}"
        lines.append(
            "| {rank} | `{name}` | {n} | {s10:.4f} | {raw} | {b10} | {partial} | {b0} | {ext:.3f} | {num:.3f} | `{stage}` | `{train}` | `{prompt}` | `{loss}` | `{sampling}` | `{dedup}` | `{decoding}` | {max_train} | {max_len} | {new_tok} | {lr} | {runtime} |".format(
                rank=rank,
                name=run.get("name"),
                n=m.get("n"),
                s10=float(m.get("score_10") or 0.0),
                raw=raw,
                b10=m.get("bucket_10_count"),
                partial=m.get("partial_count"),
                b0=m.get("bucket_0_count"),
                ext=float(m.get("extractable_rate") or 0.0),
                num=float(m.get("numeric_pair_rate") or 0.0),
                stage=cfg.get("train_stage", ""),
                train=cfg.get("train_style", ""),
                prompt=cfg.get("prompt_style", ""),
                loss=cfg.get("loss_style", ""),
                sampling=cfg.get("sampling_style", ""),
                dedup=cfg.get("dedup_train_questions", ""),
                decoding=cfg.get("decoding_style", ""),
                max_train=cfg.get("max_train_samples", ""),
                max_len=cfg.get("max_length", ""),
                new_tok=cfg.get("max_new_tokens", ""),
                lr=cfg.get("lr", ""),
                runtime=runtime_text,
            )
        )

    lines += ["", "## Best By Validation Size", ""]
    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_n[int(run.get("metrics", {}).get("n") or 0)].append(run)
    for n, group in sorted(by_n.items()):
        if not n:
            continue
        best = sorted(group, key=score_key, reverse=True)[0]
        bm = best.get("metrics", {})
        lines.append(
            f"- n={n}: `{best['name']}` score_10={float(bm.get('score_10') or 0.0):.4f}, "
            f"10-count={bm.get('bucket_10_count')}, extractable_rate={float(bm.get('extractable_rate') or 0.0):.3f}"
        )

    if ranked:
        best = ranked[0]
        cfg = best.get("config", {})
        risky = cfg.get("train_style") in {"answer_stub", "answer_only", "answer_focused"}
        lines += [
            "",
            "## Current Best Candidate",
            "",
            f"- Run: `{best['name']}`",
            f"- Path: `{best['path']}`",
            f"- Config: `{best.get('config', {})}`",
            f"- Metric-optimized answer stub risk: `{risky}`",
            "",
            "This ranking is based on local validation only. Use it to choose the next Kaggle candidate, not as a final guarantee.",
        ]

    lines += ["", "## Diagnostics By Run"]
    for run in ranked:
        d = run.get("diagnostics", {})
        lines += [
            "",
            f"### {run['name']}",
            "",
            f"- non_extractable: {d.get('non_extractable_count')}",
            f"- extractable_not_numeric: {d.get('extractable_not_numeric_count')}",
            f"- extractable_wrong: {d.get('extractable_wrong_count')}",
            f"- quoted_answer: {d.get('quoted_answer_count')}",
            f"- repetitive: {d.get('repetitive_count')}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/experiments")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    runs = []
    for child in sorted(root.iterdir() if root.exists() else []):
        if not child.is_dir():
            continue
        run = summarize_run(child, max_examples=args.max_examples)
        if run is not None:
            runs.append(run)

    best_by_validation_size = {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[int(run.get("metrics", {}).get("n") or 0)].append(run)
    for n, group in sorted(grouped.items()):
        if n:
            best_by_validation_size[str(n)] = sorted(group, key=score_key, reverse=True)[0]

    payload = {
        "root": str(root),
        "n_runs": len(runs),
        "best_by_validation_size": best_by_validation_size,
        "runs": sorted(runs, key=score_key, reverse=True),
    }

    out_json = Path(args.out_json) if args.out_json else root / "experiment_summary.json"
    out_md = Path(args.out_md) if args.out_md else root / "experiment_summary.md"
    write_json(payload, out_json)
    out_md.write_text(make_markdown(runs), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    if runs:
        best = sorted(runs, key=score_key, reverse=True)[0]
        print("Best:", best["name"], "score_10=", best["summary"].get("score_10"), best["summary"].get("buckets"))


if __name__ == "__main__":
    main()
