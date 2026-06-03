#!/usr/bin/env python3
"""Patch the final notebook config and export result artifacts.

This script does not run training. It edits the workspace notebook so a future
Kaggle execution records the final single-run config, train-query dedup, forced
decoding sweep, and report artifacts. It also parses the notebook's existing
outputs to create local figures/tables for the currently executed run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vn_gpt2_math.reporting import export_notebook_artifacts

DEFAULT_NOTEBOOK = ROOT / "notebooks" / "final_experiment.ipynb"
DEFAULT_SOURCE = Path("/Users/khoatran/Downloads/rewind(1).ipynb")
DEFAULT_ARTIFACT_DIR = ROOT / "reports" / "final_results" / "rewind_current_33p64"


def read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source or "\n")
        except SyntaxError as exc:
            raise SyntaxError(f"Notebook cell {index} has invalid Python: {exc}") from exc
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def cell_source(notebook: dict[str, Any], index: int) -> str:
    return "".join(notebook["cells"][index].get("source", []))


def set_cell_source(notebook: dict[str, Any], index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source.splitlines(True)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Patch anchor not found: {label}")
    return source.replace(old, new, 1)


def replace_once_idempotent(source: str, old: str, new: str, label: str) -> str:
    if old in source:
        return source.replace(old, new, 1)
    if new in source:
        return source
    raise RuntimeError(f"Patch anchor not found: {label}")


def replace_regex(source: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"Patch regex did not match exactly once: {label}")
    return updated


def patch_config_cell(source: str) -> str:
    source = replace_once_idempotent(
        source,
        "MAX_VALID_SAMPLES = 500 if FAST_DEV else None",
        "MAX_VALID_SAMPLES = 1000 if FAST_DEV else None",
        "valid sample count",
    )
    source = replace_once_idempotent(source, "SC_NUM_SAMPLES = 9", "SC_NUM_SAMPLES = 21", "sc sample count")
    source = replace_once_idempotent(source, "SC_TEMPERATURE = 0.4", "SC_TEMPERATURE = 0.5", "sc temperature")
    source = replace_once_idempotent(source, "SC_TOP_K = 40", "SC_TOP_K = 50", "sc top_k")
    source = replace_once_idempotent(source, "SC_TOP_P = 0.90", "SC_TOP_P = 0.95", "sc top_p")

    decoding_block = """DECODING_GRID = [
    ("sc15_t04", dict(use_beam=False, sc=True, n=15, temp=0.4, top_k=40, top_p=0.90)),
    ("sc21_t04", dict(use_beam=False, sc=True, n=21, temp=0.4, top_k=40, top_p=0.90)),
    ("sc21_t05", dict(use_beam=False, sc=True, n=21, temp=0.5, top_k=50, top_p=0.95)),
    ("sc31_t05", dict(use_beam=False, sc=True, n=31, temp=0.5, top_k=50, top_p=0.95)),
]
FORCE_DECODING_SWEEP = True
DECODING_SWEEP_SKIP_AFTER_MIN = 100000.0

# Train-only Vietnamese query dedup. Exact canonical duplicates are always removed.
# Near duplicates are removed only when their numeric signature matches, so same
# template/different-number problems are preserved.
DEDUP_TRAIN_QUERIES = True
DEDUP_SIMILAR_QUERIES = True
DEDUP_JACCARD_THRESHOLD = 0.90
DEDUP_SEQUENCE_THRESHOLD = 0.94
DEDUP_REQUIRE_NUMBER_SIGNATURE_MATCH = True
DEDUP_MIN_TOKEN_COUNT = 6
DEDUP_AUDIT_EXAMPLES = 12
DEDUP_TYPE_PRIORITY = {
    "GSM_Rephrased": 0,
    "GSM_AnsAug": 1,
    "MATH_Rephrased": 2,
    "MATH_AnsAug": 3,
    "GSM_FOBAR": 4,
    "MATH_FOBAR": 5,
    "GSM_SV": 6,
    "MATH_SV": 7,
}
TRAIN_DEDUP_REPORT = {}
"""
    if "DEDUP_TRAIN_QUERIES = True" not in source:
        source = replace_regex(
            source,
            r"DECODING_GRID = \[.*?\]\nTYPE_ROUTING_USED = False",
            decoding_block + "TYPE_ROUTING_USED = False",
            "decoding grid and dedup config",
        )
    if '"force_decoding_sweep": FORCE_DECODING_SWEEP' not in source:
        source = replace_once(
            source,
            '''    "anchor_weight": ANCHOR_WEIGHT,
    "max_new_tokens": MAX_NEW_TOKENS,
})''',
            '''    "anchor_weight": ANCHOR_WEIGHT,
    "eos_weight": EOS_WEIGHT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "max_valid_samples": MAX_VALID_SAMPLES,
    "sc_num_samples": SC_NUM_SAMPLES,
    "sc_temperature": SC_TEMPERATURE,
    "sc_top_k": SC_TOP_K,
    "sc_top_p": SC_TOP_P,
    "force_decoding_sweep": FORCE_DECODING_SWEEP,
    "dedup_train_queries": DEDUP_TRAIN_QUERIES,
    "dedup_similar_queries": DEDUP_SIMILAR_QUERIES,
    "dedup_jaccard_threshold": DEDUP_JACCARD_THRESHOLD,
    "dedup_sequence_threshold": DEDUP_SEQUENCE_THRESHOLD,
    "decoding_grid": [label for label, _ in DECODING_GRID],
})''',
            "hpo print config",
        )
    source = source.replace("# === HPO CONFIG ===", "# === FINAL SINGLE-RUN CONFIG ===")
    source = source.replace("# === END HPO CONFIG ===", "# === END FINAL SINGLE-RUN CONFIG ===")
    return source


DEDUP_CELL = r'''import difflib
import random
from collections import defaultdict

print("Loading tokenizer for length filters...")
temp_tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
temp_tok.pad_token_id = SAFE_EOS_ID
temp_tok.eos_token_id = SAFE_EOS_ID


def _strip_vietnamese_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def canonicalize_query_for_dedup(query: str) -> str:
    """Canonical form for duplicate detection only; it does not edit the training record."""
    text = unicodedata.normalize("NFKC", str(query or ""))
    text = normalize_vietnamese_number_words(text)
    text = clean_math_text(text)
    text = _strip_vietnamese_diacritics(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"[^0-9a-z+\-*/=<>.% ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedup_tokens(canon: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?|[a-z]+|[+\-*/=<>]", canon)


def _dedup_number_signature(canon: str) -> tuple[str, ...]:
    return tuple(re.findall(r"-?\d+(?:\.\d+)?", canon))


def _dedup_features(tokens: list[str]) -> set[str]:
    feats = {f"u:{tok}" for tok in tokens}
    feats.update(f"b:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return feats


def _simhash64(features: set[str]) -> int:
    if not features:
        return 0
    weights = [0] * 64
    for feat in features:
        h = int(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, value in enumerate(weights):
        if value >= 0:
            out |= 1 << bit
    return out


def _dedup_keep_key(rec: dict, idx: int) -> tuple[int, int, int, int]:
    rtype = rec.get("type") or "UNKNOWN"
    type_rank = DEDUP_TYPE_PRIORITY.get(rtype, 99)
    target_len = len(str(rec.get("_target") or rec.get("response_vi") or "").split())
    query_len = len(str(rec.get("query_vi") or "").split())
    return (type_rank, target_len, query_len, idx)


def deduplicate_train_queries(records: list[dict]) -> tuple[list[dict], dict]:
    report = {
        "enabled": bool(DEDUP_TRAIN_QUERIES),
        "input_records": len(records),
        "kept_records": len(records),
        "dropped_exact": 0,
        "dropped_similar": 0,
        "jaccard_threshold": DEDUP_JACCARD_THRESHOLD,
        "sequence_threshold": DEDUP_SEQUENCE_THRESHOLD,
        "require_number_signature_match": DEDUP_REQUIRE_NUMBER_SIGNATURE_MATCH,
        "type_counts_before": dict(Counter(r.get("type") or "UNKNOWN" for r in records)),
        "type_counts_after": {},
        "examples": [],
    }
    if not DEDUP_TRAIN_QUERIES:
        return records, report

    ranked = sorted(enumerate(records), key=lambda item: _dedup_keep_key(item[1], item[0]))
    exact_seen: dict[str, int] = {}
    buckets: dict[tuple[tuple[str, ...], int, int], list[int]] = defaultdict(list)
    kept_entries: list[tuple[int, dict, dict]] = []
    dropped_original_indices: set[int] = set()

    for original_idx, rec in ranked:
        canon = canonicalize_query_for_dedup(rec.get("query_vi", ""))
        if not canon:
            kept_entries.append((original_idx, rec, {"canon": canon, "features": set(), "numbers": tuple()}))
            continue

        exact_match = exact_seen.get(canon)
        if exact_match is not None:
            report["dropped_exact"] += 1
            dropped_original_indices.add(original_idx)
            if len(report["examples"]) < DEDUP_AUDIT_EXAMPLES:
                kept_idx, kept_rec, _ = kept_entries[exact_match]
                report["examples"].append({
                    "mode": "exact",
                    "dropped_original_idx": original_idx,
                    "dropped_type": rec.get("type"),
                    "kept_original_idx": kept_idx,
                    "kept_type": kept_rec.get("type"),
                    "query_preview": str(rec.get("query_vi", ""))[:220],
                })
            continue

        tokens = _dedup_tokens(canon)
        features = _dedup_features(tokens)
        numbers = _dedup_number_signature(canon)
        matched_idx = None
        matched_score = None

        can_check_similar = (
            DEDUP_SIMILAR_QUERIES
            and len(tokens) >= DEDUP_MIN_TOKEN_COUNT
            and (numbers or not DEDUP_REQUIRE_NUMBER_SIGNATURE_MATCH)
        )
        if can_check_similar:
            sig = _simhash64(features)
            candidate_indices: set[int] = set()
            for band in range(4):
                band_value = (sig >> (band * 16)) & 0xFFFF
                candidate_indices.update(buckets.get((numbers, band, band_value), []))

            for cand_idx in candidate_indices:
                _, cand_rec, cand_meta = kept_entries[cand_idx]
                if DEDUP_REQUIRE_NUMBER_SIGNATURE_MATCH and cand_meta["numbers"] != numbers:
                    continue
                union = features | cand_meta["features"]
                jaccard = len(features & cand_meta["features"]) / max(1, len(union))
                seq_ratio = difflib.SequenceMatcher(None, canon, cand_meta["canon"]).ratio()
                if jaccard >= DEDUP_JACCARD_THRESHOLD or seq_ratio >= DEDUP_SEQUENCE_THRESHOLD:
                    matched_idx = cand_idx
                    matched_score = {"jaccard": round(jaccard, 4), "sequence_ratio": round(seq_ratio, 4)}
                    break

        if matched_idx is not None:
            report["dropped_similar"] += 1
            dropped_original_indices.add(original_idx)
            if len(report["examples"]) < DEDUP_AUDIT_EXAMPLES:
                kept_idx, kept_rec, _ = kept_entries[matched_idx]
                report["examples"].append({
                    "mode": "similar",
                    "dropped_original_idx": original_idx,
                    "dropped_type": rec.get("type"),
                    "kept_original_idx": kept_idx,
                    "kept_type": kept_rec.get("type"),
                    "scores": matched_score,
                    "numbers": numbers,
                    "query_preview": str(rec.get("query_vi", ""))[:220],
                })
            continue

        entry_idx = len(kept_entries)
        meta = {"canon": canon, "features": features, "numbers": numbers}
        kept_entries.append((original_idx, rec, meta))
        exact_seen[canon] = entry_idx
        if features:
            sig = _simhash64(features)
            for band in range(4):
                band_value = (sig >> (band * 16)) & 0xFFFF
                buckets[(numbers, band, band_value)].append(entry_idx)

    kept_by_original_order = [rec for original_idx, rec, _ in sorted(kept_entries, key=lambda item: item[0])]
    report["kept_records"] = len(kept_by_original_order)
    report["type_counts_after"] = dict(Counter(r.get("type") or "UNKNOWN" for r in kept_by_original_order))
    report["dropped_total"] = len(dropped_original_indices)
    report["drop_rate"] = report["dropped_total"] / max(1, report["input_records"])
    return kept_by_original_order, report


print(f"Starting pipeline with {len(raw_train_records)} raw train records.")

allowed_unconditionally = {"GSM_AnsAug", "GSM_Rephrased", "MATH_AnsAug", "MATH_Rephrased"}
verbose_tracks = {"GSM_SV", "GSM_FOBAR", "MATH_SV", "MATH_FOBAR"}

MAX_VERBOSE_TOKENS = 100
VERBOSE_KEEP_PROB = 0.15

filtered_train_records = []
sv_fobar_kept_short = 0
sv_fobar_kept_verbose = 0
sv_fobar_dropped = 0

for r in raw_train_records:
    rtype = r.get("type")

    if rtype in allowed_unconditionally:
        filtered_train_records.append(r)
    elif rtype in verbose_tracks:
        response_text = r.get("response_vi", "")
        tok_len = len(temp_tok(response_text, add_special_tokens=False)["input_ids"])

        if tok_len <= MAX_VERBOSE_TOKENS:
            filtered_train_records.append(r)
            sv_fobar_kept_short += 1
        elif random.random() < VERBOSE_KEEP_PROB:
            filtered_train_records.append(r)
            sv_fobar_kept_verbose += 1
        else:
            sv_fobar_dropped += 1

print(f"Verbose Tracks -> Kept Short: {sv_fobar_kept_short} | Kept Verbose (Diversity): {sv_fobar_kept_verbose} | Dropped: {sv_fobar_dropped}")

raw_train_records = filtered_train_records

if MAX_TRAIN_SAMPLES:
    raw_train_records = raw_train_records[:MAX_TRAIN_SAMPLES]
if MAX_VALID_SAMPLES:
    valid_records = valid_records[:MAX_VALID_SAMPLES]
'''


REPORT_CELL = r'''# ============================================================
# Report artifacts
# ============================================================
import csv

REPORT_ARTIFACT_DIR = Path("/kaggle/working/report_artifacts")
REPORT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trainer_epoch_losses(log_history: list[dict]) -> list[dict]:
    rows = []
    for item in log_history:
        if "eval_loss" not in item or "epoch" not in item:
            continue
        epoch = int(round(float(item["epoch"])))
        train_loss = None
        for prev in reversed(log_history[: log_history.index(item)]):
            if "loss" in prev:
                train_loss = prev["loss"]
                break
        rows.append({
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": item.get("eval_loss"),
        })
    return rows


loss_rows = _trainer_epoch_losses(globals().get("TRAINER_LOG_HISTORY", []))
epoch_rows = []
for row in hpo_report.get("epoch_scores", []):
    merged = {
        "epoch": row["epoch"],
        "checkpoint": Path(row["checkpoint"]).name,
        "raw_score": row["raw_score"],
        "exact_count": row["exact_count"],
        "extractable_count": row["extractable_count"],
        "score_pct": row["raw_score"] / max(1, valid_result["summary"]["max_raw_score"]),
    }
    for loss_row in loss_rows:
        if loss_row["epoch"] == row["epoch"]:
            merged.update(loss_row)
            break
    epoch_rows.append(merged)

decoding_rows = []
for row in hpo_report.get("decoding_sweep", []):
    cfg = row.get("config", {})
    decoding_rows.append({
        "label": row["label"],
        "raw_score": row["raw_score"],
        "exact_count": row["exact_count"],
        "extractable_count": row["extractable_count"],
        "n": cfg.get("n"),
        "temperature": cfg.get("temp"),
        "top_k": cfg.get("top_k"),
        "top_p": cfg.get("top_p"),
    })

type_rows = []
for label, row in valid_result["summary"].get("score_by_type", {}).items():
    type_rows.append({
        "type": label,
        "count": row["n"],
        "mean_score": row["mean_score"],
        "exact_count": row["exact_count"],
        "raw_score": row["raw_score"],
        "extractable_rate": row["extractable_rate"],
    })

bucket_rows = [
    {"bucket": str(k), "count": v}
    for k, v in valid_result["summary"].get("buckets", {}).items()
]

_write_csv(REPORT_ARTIFACT_DIR / "epoch_metrics.csv", epoch_rows)
_write_csv(REPORT_ARTIFACT_DIR / "decoding_sweep.csv", decoding_rows)
_write_csv(REPORT_ARTIFACT_DIR / "type_scores.csv", type_rows)
_write_csv(REPORT_ARTIFACT_DIR / "bucket_distribution.csv", bucket_rows)
(REPORT_ARTIFACT_DIR / "training_config.json").write_text(
    json.dumps(hpo_report["training_config"], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(REPORT_ARTIFACT_DIR / "dedup_report.json").write_text(
    json.dumps(TRAIN_DEDUP_REPORT, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

try:
    import matplotlib.pyplot as plt

    if epoch_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([r["epoch"] for r in epoch_rows], [r["raw_score"] for r in epoch_rows], marker="o")
        ax.set_title("Epoch Checkpoint Validation Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Raw validation score")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORT_ARTIFACT_DIR / "epoch_checkpoint_score.png", dpi=200)
        fig.savefig(REPORT_ARTIFACT_DIR / "epoch_checkpoint_score.svg")
        plt.close(fig)

        if any(r.get("validation_loss") is not None for r in epoch_rows):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([r["epoch"] for r in epoch_rows], [r.get("training_loss") for r in epoch_rows], marker="o", label="Training loss")
            ax.plot([r["epoch"] for r in epoch_rows], [r.get("validation_loss") for r in epoch_rows], marker="o", label="Validation loss")
            ax.set_title("Teacher-Forced Loss by Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(REPORT_ARTIFACT_DIR / "epoch_loss_curve.png", dpi=200)
            fig.savefig(REPORT_ARTIFACT_DIR / "epoch_loss_curve.svg")
            plt.close(fig)

    if type_rows:
        type_rows_sorted = sorted(type_rows, key=lambda r: r["mean_score"], reverse=True)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar([r["type"] for r in type_rows_sorted], [r["mean_score"] for r in type_rows_sorted])
        ax.set_title("Validation Score by Sub-Track")
        ax.set_xlabel("Sub-track")
        ax.set_ylabel("Mean score / 10")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(REPORT_ARTIFACT_DIR / "subtrack_mean_score.png", dpi=200)
        fig.savefig(REPORT_ARTIFACT_DIR / "subtrack_mean_score.svg")
        plt.close(fig)

    if decoding_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([r["label"] for r in decoding_rows], [r["raw_score"] for r in decoding_rows])
        ax.set_title("Decoding Sweep")
        ax.set_xlabel("Config")
        ax.set_ylabel("Raw validation score")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(REPORT_ARTIFACT_DIR / "decoding_sweep.png", dpi=200)
        fig.savefig(REPORT_ARTIFACT_DIR / "decoding_sweep.svg")
        plt.close(fig)
except Exception as exc:
    print("Matplotlib artifact generation skipped:", repr(exc))

results_md = f"""# Results Summary

Final validation score: {valid_result['summary']['raw_score']} / {valid_result['summary']['max_raw_score']} ({valid_result['summary']['score_pct'] * 100:.2f}%).

Best epoch: {hpo_report['best_epoch']} using checkpoint `{Path(best_ckpt).name}`.

Best decoding config: `{hpo_report['best_decoding_config']}` with parameters `{best_decoding_config}`.

Exact matches: {valid_result['summary']['exact_count']} / {valid_result['summary']['n']}.

Train dedup: kept {TRAIN_DEDUP_REPORT.get('kept_records', 'NA')} of {TRAIN_DEDUP_REPORT.get('input_records', 'NA')} records; exact drops={TRAIN_DEDUP_REPORT.get('dropped_exact', 'NA')}, similar drops={TRAIN_DEDUP_REPORT.get('dropped_similar', 'NA')}.

Artifacts saved in `{REPORT_ARTIFACT_DIR}`.
"""
(REPORT_ARTIFACT_DIR / "results_section.md").write_text(results_md, encoding="utf-8")
print("Saved report artifacts to", REPORT_ARTIFACT_DIR)
'''


def patch_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    set_cell_source(notebook, 2, patch_config_cell(cell_source(notebook, 2)))
    set_cell_source(notebook, 16, DEDUP_CELL)

    source17 = cell_source(notebook, 17)
    if "After Vietnamese query dedup" not in source17:
        source17 = replace_once(
            source17,
            '''train_records = [r for r in train_records if is_valid_length(r, temp_tok, max_total_length=MAX_LENGTH)]
print(f"   -> After total length check: {len(train_records)}")
# Sanity check: log distribution của response lengths sau filter
''',
            '''train_records = [r for r in train_records if is_valid_length(r, temp_tok, max_total_length=MAX_LENGTH)]
print(f"   -> After total length check: {len(train_records)}")

if DEDUP_TRAIN_QUERIES:
    before_dedup = len(train_records)
    train_records, TRAIN_DEDUP_REPORT = deduplicate_train_queries(train_records)
    print(
        f"   -> After Vietnamese query dedup: {len(train_records)}/{before_dedup} "
        f"(exact_drop={TRAIN_DEDUP_REPORT['dropped_exact']}, "
        f"similar_drop={TRAIN_DEDUP_REPORT['dropped_similar']}, "
        f"drop_rate={TRAIN_DEDUP_REPORT['drop_rate']*100:.1f}%)"
    )
    if TRAIN_DEDUP_REPORT.get("examples"):
        print("   -> Dedup audit examples:")
        print(json.dumps(TRAIN_DEDUP_REPORT["examples"][:3], ensure_ascii=False, indent=2))
else:
    TRAIN_DEDUP_REPORT = {
        "enabled": False,
        "input_records": len(train_records),
        "kept_records": len(train_records),
        "dropped_exact": 0,
        "dropped_similar": 0,
        "dropped_total": 0,
        "drop_rate": 0.0,
        "type_counts_before": dict(Counter(r.get("type") or "UNKNOWN" for r in train_records)),
        "type_counts_after": dict(Counter(r.get("type") or "UNKNOWN" for r in train_records)),
        "examples": [],
    }
# Sanity check: log distribution của response lengths sau filter
''',
            "dedup after length filter",
        )
    set_cell_source(notebook, 17, source17)

    source31 = cell_source(notebook, 31)
    if "TRAINER_LOG_HISTORY = trainer.state.log_history" not in source31:
        source31 = replace_once(
            source31,
            "train_elapsed_min = train_dt / 60\nSKIP_DECODING_SWEEP = train_elapsed_min > 120",
            "train_elapsed_min = train_dt / 60\nTRAINER_LOG_HISTORY = trainer.state.log_history\nSKIP_DECODING_SWEEP = False if FORCE_DECODING_SWEEP else train_elapsed_min > DECODING_SWEEP_SKIP_AFTER_MIN",
            "forced decoding sweep guard",
        )
    if "Decoding sweep forced on; runtime guard disabled" not in source31:
        source31 = replace_once(
            source31,
            '''if SKIP_DECODING_SWEEP:
    print(f"WARNING: Training took {train_elapsed_min:.1f}min. Skipping decoding sweep.")
''',
            '''if SKIP_DECODING_SWEEP:
    print(f"WARNING: Training took {train_elapsed_min:.1f}min. Skipping decoding sweep.")
elif FORCE_DECODING_SWEEP:
    print("Decoding sweep forced on; runtime guard disabled for this final run.")
''',
            "forced sweep message",
        )
    set_cell_source(notebook, 31, source31)

    source33 = cell_source(notebook, 33)
    source33 = source33.replace(
        '''        "max_length": MAX_LENGTH,
        "batch_size": PER_DEVICE_BATCH_SIZE,
''',
        '''        "max_length": MAX_LENGTH,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_valid_samples": MAX_VALID_SAMPLES,
        "batch_size": PER_DEVICE_BATCH_SIZE,
''',
    )
    source33 = source33.replace(
        '''        "computation_weight": COMPUTATION_WEIGHT,
        "anchor_weight": ANCHOR_WEIGHT,
        "seed": SEED,
''',
        '''        "computation_weight": COMPUTATION_WEIGHT,
        "anchor_weight": ANCHOR_WEIGHT,
        "eos_weight": EOS_WEIGHT,
        "seed": SEED,
        "dedup_train_queries": DEDUP_TRAIN_QUERIES,
        "dedup_similar_queries": DEDUP_SIMILAR_QUERIES,
        "dedup_jaccard_threshold": DEDUP_JACCARD_THRESHOLD,
        "dedup_sequence_threshold": DEDUP_SEQUENCE_THRESHOLD,
        "train_dedup_report": TRAIN_DEDUP_REPORT,
        "force_decoding_sweep": FORCE_DECODING_SWEEP,
        "decoding_grid": [{"label": label, "config": config} for label, config in DECODING_GRID],
''',
    )
    source33 = source33.replace(
        '''    "decoding_sweep": [],
    "best_decoding_config": None,
''',
        '''    "decoding_sweep": [],
    "decoding_grid": [{"label": label, "config": config} for label, config in DECODING_GRID],
    "best_decoding_config": None,
''',
    )
    source33 = source33.replace(
        '''best_decoding_label = "sc9_t04"
best_decoding_config = dict(use_beam=False, sc=True, n=9, temp=0.4, top_k=40, top_p=0.90)
''',
        '''best_decoding_label = "sc21_t05"
best_decoding_config = dict(use_beam=False, sc=True, n=21, temp=0.5, top_k=50, top_p=0.95)
''',
    )
    source33 = source33.replace(
        '''if SKIP_DECODING_SWEEP:
    print("\\nSkipping decoding sweep due runtime guard; using sc9_t04.")
''',
        '''if SKIP_DECODING_SWEEP:
    print("\\nSkipping decoding sweep due runtime guard; using sc21_t05 fallback.")
''',
    )
    source33 = source33.replace(
        '''    "type_routing_used": bool(TYPE_ROUTING_USED),
}
''',
        '''    "type_routing_used": bool(TYPE_ROUTING_USED),
    "train_dedup_report": TRAIN_DEDUP_REPORT,
}
''',
    )
    set_cell_source(notebook, 33, source33)

    if not any("Report artifacts" in cell_source(notebook, i) for i in range(len(notebook["cells"]))):
        notebook["cells"].insert(
            34,
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": REPORT_CELL.splitlines(True),
            },
        )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--skip-patch", action="store_true")
    args = parser.parse_args()

    if not args.notebook.exists():
        if not args.source.exists():
            raise FileNotFoundError(f"Neither notebook nor source exists: {args.notebook} | {args.source}")
        args.notebook.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source, args.notebook)

    notebook = read_notebook(args.notebook)
    if not args.skip_patch:
        notebook = patch_notebook(notebook)
        write_notebook(args.notebook, notebook)

    summary = export_notebook_artifacts(notebook, args.artifact_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
