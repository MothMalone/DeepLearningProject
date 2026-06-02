"""Validation scoring and diagnostics for the competition metric."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .answers import extract_gold, extract_pred, parse_number, rel_error, score_one


def align_predictions_with_gold(pred_items: list[dict], gold_items: list[dict]) -> list[tuple[dict, dict]]:
    pred_has_id = all("id" in x for x in pred_items)
    gold_has_id = all("id" in x for x in gold_items)

    if pred_has_id and gold_has_id:
        pred_map = {str(x["id"]): x for x in pred_items}
        pairs = []
        missing = []
        for gold in gold_items:
            gid = str(gold["id"])
            if gid in pred_map:
                pairs.append((pred_map[gid], gold))
            else:
                missing.append(gid)
        if missing:
            raise ValueError(f"Predictions missing {len(missing)} ids, e.g. {missing[:5]}")
        return pairs

    if len(pred_items) != len(gold_items):
        raise ValueError(f"Prediction count {len(pred_items)} != gold count {len(gold_items)}")
    return list(zip(pred_items, gold_items))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_rel_error_diagnostics(rows: list[dict]) -> dict[str, Any]:
    errors = []
    for row in rows:
        gold = row.get("gold_num")
        pred = row.get("pred_num")
        if gold is None or pred is None:
            continue
        try:
            errors.append(abs(float(pred) - float(gold)) / max(1.0, abs(float(gold))))
        except (TypeError, ValueError):
            continue

    if not errors:
        return {}

    median = _percentile(errors, 50)
    cap = median * 10
    return {
        "n_with_error": len(errors),
        "mean_rel_error": float(sum(errors) / len(errors)),
        "median_rel_error": float(median),
        "p90_rel_error": float(_percentile(errors, 90)),
        "p95_rel_error": float(_percentile(errors, 95)),
        "p99_rel_error": float(_percentile(errors, 99)),
        "capped_mean_rel_error": float(sum(min(x, cap) for x in errors) / len(errors)),
        "worst_5_errors": [float(x) for x in sorted(errors, reverse=True)[:5]],
    }


def compute_score_by_type(rows: list[dict]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "raw_score": 0,
            "exact_count": 0,
            "extractable": 0,
            "numeric_pairs": 0,
            "buckets": {"10": 0, "5": 0, "1": 0, "0": 0},
        }
    )
    for row in rows:
        typ = row.get("type") or "UNKNOWN"
        item = grouped[typ]
        score = int(row.get("score") or 0)
        item["n"] += 1
        item["raw_score"] += score
        item["exact_count"] += int(score == 10)
        item["extractable"] += int(bool(row.get("extractable")))
        item["numeric_pairs"] += int(row.get("gold_num") is not None and row.get("pred_num") is not None)
        item["buckets"][str(score)] = item["buckets"].get(str(score), 0) + 1

    result = dict(grouped)
    for item in result.values():
        n = max(1, int(item["n"]))
        item["mean_score"] = item["raw_score"] / n
        item["score_pct"] = item["raw_score"] / (10 * n)
        item["extractable_rate"] = item["extractable"] / n
        item["numeric_pair_rate"] = item["numeric_pairs"] / n
    return result


def evaluate(pred_items: list[dict], gold_items: list[dict]) -> dict[str, Any]:
    rows = []
    total = 0
    buckets = {"10": 0, "5": 0, "1": 0, "0": 0}
    extractable_count = 0
    numeric_pair_count = 0
    rel_errors = []

    for pred_rec, gold_rec in align_predictions_with_gold(pred_items, gold_items):
        gold_answer = extract_gold(gold_rec)
        pred_answer = extract_pred(pred_rec)
        is_extractable = pred_answer is not None
        extractable_count += int(is_extractable)

        gold_num = parse_number(gold_answer)
        pred_num = parse_number(pred_answer)
        error = rel_error(pred_num, gold_num)
        if gold_num is not None and pred_num is not None and error is not None:
            numeric_pair_count += 1
            rel_errors.append(error)

        score = score_one(error, is_extractable)
        total += score
        buckets[str(score)] += 1
        rows.append(
            {
                "id": gold_rec.get("id", pred_rec.get("id")),
                "type": gold_rec.get("type") or pred_rec.get("type"),
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "gold_num": gold_num,
                "pred_num": pred_num,
                "rel_error": error,
                "extractable": is_extractable,
                "score": score,
            }
        )

    n = len(rows)
    diagnostics = compute_rel_error_diagnostics(rows)
    summary: dict[str, Any] = {
        "n": n,
        "raw_score": total,
        "max_raw_score": n * 10,
        "score_10": total / n if n else 0.0,
        "score_pct": total / (n * 10) if n else 0.0,
        "extractable": extractable_count,
        "extractable_count": extractable_count,
        "numeric_pairs": numeric_pair_count,
        "numeric_pair_count": numeric_pair_count,
        "exact_count": buckets["10"],
        "buckets": buckets,
        "bucket_distribution": buckets,
        "score_by_type": compute_score_by_type(rows),
        "rel_error_mean": sum(rel_errors) / len(rel_errors) if rel_errors else None,
        "rel_error_diagnostics": diagnostics,
    }
    summary.update(diagnostics)
    return {"summary": summary, "rows": rows}
