"""Compact failure-mode diagnostics used in the technical report."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_records


class VietnameseMathErrorAnalyzer:
    """Heuristic failure taxonomy for Vietnamese math validation outputs."""

    def __init__(self, valid_report_path: str | Path, valid_output_path: str | Path):
        import json

        self.report_path = Path(valid_report_path)
        self.output_path = Path(valid_output_path)
        with self.report_path.open("r", encoding="utf-8") as f:
            self.report_data = json.load(f)
        with self.output_path.open("r", encoding="utf-8") as f:
            self.output_data = json.load(f)
        self.rows = self.report_data["rows"]
        self.summary = self.report_data["summary"]

    def classify_row(self, row: dict[str, Any], pred_item: dict[str, Any]) -> list[str]:
        modes = []
        output = pred_item.get("model_output", "")

        if not row.get("extractable"):
            modes.append("extraction_failure")

        words = output.lower().split()
        if len(words) <= 4 and row.get("score") == 0:
            modes.append("premature_eos")
        if len(words) > 10:
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            if trigrams and Counter(trigrams).most_common(1)[0][1] > 4:
                modes.append("repetition_loop")

        if row.get("extractable") and row.get("score") == 0:
            pred_num = row.get("pred_num")
            gold_num = row.get("gold_num")
            if pred_num is not None and gold_num is not None:
                if pred_num == -gold_num and gold_num != 0:
                    modes.append("operator_sign_mistake")
                elif gold_num != 0 and pred_num != 0:
                    ratio = abs(pred_num / gold_num)
                    if ratio > 0 and not np.isinf(ratio) and abs(np.log10(ratio) - round(np.log10(ratio))) < 1e-4:
                        modes.append("decimal_placement_error")
                    else:
                        modes.append("arithmetic_mistake")
                else:
                    modes.append("arithmetic_mistake")
            else:
                modes.append("formatting_drift")

        if not modes and row.get("score", 0) < 10:
            modes.append("reasoning_collapse")
        return modes or ["clean_pass"]

    def summarize(self) -> dict[str, Any]:
        failure_counts = Counter()
        type_scores: dict[str, list[int]] = {}
        for idx, row in enumerate(self.rows):
            pred = self.output_data[idx] if idx < len(self.output_data) else {}
            failure_counts.update(self.classify_row(row, pred))
            type_scores.setdefault(row.get("type") or "UNKNOWN", []).append(int(row.get("score", 0)))
        return {
            "summary": self.summary,
            "failure_counts": dict(failure_counts),
            "type_performance": {
                typ: {"count": len(scores), "mean": sum(scores) / len(scores)}
                for typ, scores in sorted(type_scores.items())
            },
        }
