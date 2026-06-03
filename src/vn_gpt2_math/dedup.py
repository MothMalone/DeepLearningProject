"""Train-query deduplication utilities."""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .targets import clean_math_text


TYPE_PRIORITY = {
    "GSM_Rephrased": 0,
    "GSM_AnsAug": 1,
    "MATH_Rephrased": 2,
    "MATH_AnsAug": 3,
    "GSM_FOBAR": 4,
    "MATH_FOBAR": 5,
    "GSM_SV": 6,
    "MATH_SV": 7,
}

WORD_NUMBERS = {
    "không": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "năm": "5",
    "sáu": "6",
    "bảy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
}


@dataclass(frozen=True)
class DedupConfig:
    similar: bool = True
    jaccard_threshold: float = 0.90
    sequence_threshold: float = 0.94
    require_number_signature_match: bool = True
    min_token_count: int = 6
    audit_examples: int = 12


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def canonicalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKC", str(query or ""))
    for word, number in WORD_NUMBERS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", number, text, flags=re.IGNORECASE)
    text = strip_diacritics(clean_math_text(text)).lower()
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"[^0-9a-z+\-*/=<>.% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_tokens(canonical: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?|[a-z]+|[+\-*/=<>]", canonical)


def number_signature(canonical: str) -> tuple[str, ...]:
    return tuple(re.findall(r"-?\d+(?:\.\d+)?", canonical))


def feature_set(tokens: list[str]) -> set[str]:
    features = {f"u:{token}" for token in tokens}
    features.update(f"b:{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    return features


def simhash64(features: set[str]) -> int:
    if not features:
        return 0
    weights = [0] * 64
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << bit
    return out


def keep_key(record: dict[str, Any], index: int) -> tuple[int, int, int, int]:
    record_type = record.get("type") or "UNKNOWN"
    target_len = len(str(record.get("_target") or record.get("response_vi") or "").split())
    query_len = len(str(record.get("query_vi") or "").split())
    return (TYPE_PRIORITY.get(record_type, 99), target_len, query_len, index)


def deduplicate_records(records: list[dict[str, Any]], config: DedupConfig | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or DedupConfig()
    ranked = sorted(enumerate(records), key=lambda item: keep_key(item[1], item[0]))
    exact_seen: dict[str, int] = {}
    buckets: dict[tuple[tuple[str, ...], int, int], list[int]] = defaultdict(list)
    kept: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    report: dict[str, Any] = {
        "config": asdict(cfg),
        "input_records": len(records),
        "kept_records": 0,
        "dropped_exact": 0,
        "dropped_similar": 0,
        "type_counts_before": dict(Counter(record.get("type") or "UNKNOWN" for record in records)),
        "type_counts_after": {},
        "examples": [],
    }

    for original_index, record in ranked:
        canonical = canonicalize_query(record.get("query_vi", ""))
        if not canonical:
            kept.append((original_index, record, {"canonical": canonical, "features": set(), "numbers": tuple()}))
            continue

        exact_index = exact_seen.get(canonical)
        if exact_index is not None:
            report["dropped_exact"] += 1
            if len(report["examples"]) < cfg.audit_examples:
                kept_index, kept_record, _ = kept[exact_index]
                report["examples"].append(
                    {
                        "mode": "exact",
                        "dropped_index": original_index,
                        "dropped_type": record.get("type"),
                        "kept_index": kept_index,
                        "kept_type": kept_record.get("type"),
                    }
                )
            continue

        tokens = query_tokens(canonical)
        features = feature_set(tokens)
        numbers = number_signature(canonical)
        match_index: int | None = None
        match_score: dict[str, float] | None = None

        if cfg.similar and len(tokens) >= cfg.min_token_count and (numbers or not cfg.require_number_signature_match):
            signature = simhash64(features)
            candidate_indices: set[int] = set()
            for band in range(4):
                candidate_indices.update(buckets.get((numbers, band, (signature >> (band * 16)) & 0xFFFF), []))
            for candidate_index in candidate_indices:
                _, _, meta = kept[candidate_index]
                if cfg.require_number_signature_match and meta["numbers"] != numbers:
                    continue
                union = features | meta["features"]
                jaccard = len(features & meta["features"]) / max(1, len(union))
                sequence = difflib.SequenceMatcher(None, canonical, meta["canonical"]).ratio()
                if jaccard >= cfg.jaccard_threshold or sequence >= cfg.sequence_threshold:
                    match_index = candidate_index
                    match_score = {"jaccard": round(jaccard, 4), "sequence": round(sequence, 4)}
                    break

        if match_index is not None:
            report["dropped_similar"] += 1
            if len(report["examples"]) < cfg.audit_examples:
                kept_index, kept_record, _ = kept[match_index]
                report["examples"].append(
                    {
                        "mode": "similar",
                        "dropped_index": original_index,
                        "dropped_type": record.get("type"),
                        "kept_index": kept_index,
                        "kept_type": kept_record.get("type"),
                        "numbers": numbers,
                        "scores": match_score,
                    }
                )
            continue

        entry_index = len(kept)
        meta = {"canonical": canonical, "features": features, "numbers": numbers}
        kept.append((original_index, record, meta))
        exact_seen[canonical] = entry_index
        if features:
            signature = simhash64(features)
            for band in range(4):
                buckets[(numbers, band, (signature >> (band * 16)) & 0xFFFF)].append(entry_index)

    output = [record for _, record, _ in sorted(kept, key=lambda item: item[0])]
    report["kept_records"] = len(output)
    report["dropped_total"] = report["dropped_exact"] + report["dropped_similar"]
    report["drop_rate"] = report["dropped_total"] / max(1, len(records))
    report["type_counts_after"] = dict(Counter(record.get("type") or "UNKNOWN" for record in output))
    return output, report
