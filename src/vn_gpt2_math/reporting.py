"""Result extraction and figure generation for experiment reports."""

from __future__ import annotations

import csv
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any


def collect_notebook_output_text(notebook: dict[str, Any]) -> str:
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                chunks.append("".join(output["text"]))
            data = output.get("data", {})
            for key in ("text/html", "text/plain"):
                if key in data:
                    value = data[key]
                    chunks.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(chunks)


def extract_training_losses(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in re.finditer(
        r"<tr>\s*<td>(\d+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>\s*</tr>",
        text,
        flags=re.S,
    ):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "training_loss": float(match.group(2)),
                "validation_loss": float(match.group(3)),
            }
        )
    return rows


def extract_epoch_scores(text: str) -> list[dict[str, Any]]:
    rows = []
    for match in re.finditer(r"Epoch\s+(\d+):\s+(checkpoint-\d+)\s+raw_score=(\d+)\s+exact=(\d+)", text):
        rows.append(
            {
                "epoch": int(match.group(1)),
                "checkpoint": match.group(2),
                "raw_score": int(match.group(3)),
                "exact_count": int(match.group(4)),
            }
        )
    return rows


def extract_decoding_scores(text: str) -> list[dict[str, Any]]:
    rows = []
    pattern = r"\[eval:([^\]]+)\]\s+raw=(\d+)/(\d+)\s+exact=(\d+)\s+extractable=(\d+)/(\d+)\s+median_re=([^\s]+)"
    for match in re.finditer(pattern, text):
        label = match.group(1)
        if label.startswith("epoch") or label in {"baseline", "final_valid"}:
            continue
        rows.append(
            {
                "label": label,
                "raw_score": int(match.group(2)),
                "max_raw_score": int(match.group(3)),
                "score_pct": int(match.group(2)) / max(1, int(match.group(3))),
                "exact_count": int(match.group(4)),
                "extractable_count": int(match.group(5)),
                "n": int(match.group(6)),
                "median_rel_error": None if match.group(7) == "None" else float(match.group(7)),
            }
        )
    return rows


def extract_json_after_marker(text: str, marker: str) -> dict[str, Any] | None:
    start = text.find(marker)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[brace:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_failure_table(text: str) -> list[dict[str, Any]]:
    rows = []
    start = text.find("### FAILURE MATRIX DAMAGE DISTRIBUTION")
    if start < 0:
        return rows
    block = text[start:].split("### SUB-TRACK", 1)[0]
    for line in block.splitlines():
        if not line.startswith("|") or "failures" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 3:
            rows.append({"failure": parts[0], "occurrences": int(float(parts[1])), "mean_score": float(parts[2])})
    return rows


def extract_type_table(text: str) -> list[dict[str, Any]]:
    rows = []
    start = text.find("### SUB-TRACK TRACKING PERFORMANCE")
    if start < 0:
        return rows
    block = text[start:].split("[EXTRACTION SIMULATOR]", 1)[0]
    for line in block.splitlines():
        if not line.startswith("|") or "count" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 3:
            rows.append({"type": parts[0], "count": int(float(parts[1])), "mean_score": float(parts[2])})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plotter(artifact_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir / ".mplconfig"))
    (artifact_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figures(
    artifact_dir: Path,
    epoch_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    decoding_rows: list[dict[str, Any]],
) -> None:
    plt = _plotter(artifact_dir)
    if epoch_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([r["epoch"] for r in epoch_rows], [r["raw_score"] for r in epoch_rows], marker="o", color="#22577a")
        best = max(epoch_rows, key=lambda row: row["raw_score"])
        ax.scatter([best["epoch"]], [best["raw_score"]], color="#d62828", zorder=3)
        ax.set_title("Epoch Checkpoint Selection")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Raw validation score")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(artifact_dir / "epoch_checkpoint_score.png", dpi=220)
        fig.savefig(artifact_dir / "epoch_checkpoint_score.svg")
        plt.close(fig)

        if "training_loss" in epoch_rows[0]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([r["epoch"] for r in epoch_rows], [r["training_loss"] for r in epoch_rows], marker="o", label="Training loss")
            ax.plot([r["epoch"] for r in epoch_rows], [r["validation_loss"] for r in epoch_rows], marker="o", label="Validation loss")
            ax.set_title("Teacher-Forced Loss by Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(artifact_dir / "epoch_loss_curve.png", dpi=220)
            fig.savefig(artifact_dir / "epoch_loss_curve.svg")
            plt.close(fig)

            fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
            ax1.plot([r["epoch"] for r in epoch_rows], [r["validation_loss"] for r in epoch_rows], marker="o", color="#8d99ae")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Validation loss", color="#8d99ae")
            ax2 = ax1.twinx()
            ax2.plot([r["epoch"] for r in epoch_rows], [r["raw_score"] for r in epoch_rows], marker="o", color="#22577a")
            ax2.set_ylabel("Raw validation score", color="#22577a")
            ax1.set_title("Validation Loss vs Generative Score")
            fig.tight_layout()
            fig.savefig(artifact_dir / "loss_vs_score.png", dpi=220)
            fig.savefig(artifact_dir / "loss_vs_score.svg")
            plt.close(fig)

    if type_rows:
        rows = sorted(type_rows, key=lambda row: row["mean_score"], reverse=True)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar([r["type"] for r in rows], [r["mean_score"] for r in rows], color="#457b9d")
        ax.set_title("Sub-Track Validation Performance")
        ax.set_xlabel("Sub-track")
        ax.set_ylabel("Mean score / 10")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(artifact_dir / "subtrack_mean_score.png", dpi=220)
        fig.savefig(artifact_dir / "subtrack_mean_score.svg")
        plt.close(fig)

    if failure_rows:
        rows = sorted(failure_rows, key=lambda row: row["occurrences"], reverse=True)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar([r["failure"] for r in rows], [r["occurrences"] for r in rows], color="#6a994e")
        ax.set_title("Failure Mode Distribution")
        ax.set_xlabel("Failure mode")
        ax.set_ylabel("Occurrences")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(artifact_dir / "failure_mode_distribution.png", dpi=220)
        fig.savefig(artifact_dir / "failure_mode_distribution.svg")
        plt.close(fig)

    if decoding_rows:
        rows = sorted(decoding_rows, key=lambda row: row["raw_score"], reverse=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#d62828" if i == 0 else "#457b9d" for i, _ in enumerate(rows)]
        ax.bar([r["label"] for r in rows], [r["raw_score"] for r in rows], color=colors)
        ax.set_title("Decoding Strategy Comparison")
        ax.set_xlabel("Decoding config")
        ax.set_ylabel("Raw validation score")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(artifact_dir / "decoding_sweep.png", dpi=220)
        fig.savefig(artifact_dir / "decoding_sweep.svg")
        plt.close(fig)


def export_notebook_artifacts(notebook: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text = collect_notebook_output_text(notebook)
    losses = extract_training_losses(text)
    epoch_rows = extract_epoch_scores(text)
    decoding_rows = extract_decoding_scores(text)
    failure_rows = extract_failure_table(text)
    type_rows = extract_type_table(text)
    final_summary = extract_json_after_marker(text, "Wrote /kaggle/working/valid_output.json")

    loss_by_epoch = {row["epoch"]: row for row in losses}
    for row in epoch_rows:
        if row["epoch"] in loss_by_epoch:
            row.update(loss_by_epoch[row["epoch"]])
            row["loss_gap"] = row["validation_loss"] - row["training_loss"]
        if final_summary:
            row["score_pct"] = row["raw_score"] / max(1, final_summary.get("max_raw_score", 1))

    write_csv(artifact_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(artifact_dir / "decoding_sweep.csv", decoding_rows)
    write_csv(artifact_dir / "subtrack_metrics.csv", type_rows)
    write_csv(artifact_dir / "failure_matrix.csv", failure_rows)
    if final_summary:
        (artifact_dir / "validation_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(artifact_dir / "bucket_distribution.csv", [{"bucket": str(k), "count": v} for k, v in final_summary.get("buckets", {}).items()])

    save_figures(artifact_dir, epoch_rows, type_rows, failure_rows, decoding_rows)

    best_epoch = max(epoch_rows, key=lambda row: row["raw_score"]) if epoch_rows else None
    score_line = "unknown"
    if final_summary:
        score_line = f"{final_summary['raw_score']} / {final_summary['max_raw_score']} ({final_summary['score_pct'] * 100:.2f}%)"
    results = textwrap.dedent(
        f"""\
        # Results Summary

        Final validation score: {score_line}.

        Best checkpoint epoch: {best_epoch['epoch'] if best_epoch else 'NA'}.

        Highest-scoring sub-track: {max(type_rows, key=lambda r: r['mean_score'])['type'] if type_rows else 'NA'}.

        Main failure mode: {failure_rows[0]['failure'] if failure_rows else 'NA'}.
        """
    )
    (artifact_dir / "results_section.md").write_text(results, encoding="utf-8")

    return {
        "epoch_rows": len(epoch_rows),
        "decoding_rows": len(decoding_rows),
        "type_rows": len(type_rows),
        "failure_rows": len(failure_rows),
        "final_summary": final_summary,
        "artifact_dir": str(artifact_dir),
    }
