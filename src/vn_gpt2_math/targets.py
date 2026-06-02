"""Final rewind3-style preprocessing and target construction."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .answers import VI_ANCHORS, extract_answer, parse_number
from .config import PROMPT_TEMPLATE

VIET_DIGIT_MAP: dict[str, int] = {
    "không": 0,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
}


def clean_math_text(text: str) -> str:
    """Light typographical cleanup used by the final notebook."""
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1/\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\times", " * ").replace("\\cdot", " * ")
    text = re.sub(r"\$([^$]{1,100})\$", r"\1", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_computation_chain(response_vi: str) -> Optional[str]:
    """Extract compact arithmetic equations from the gold Vietnamese solution."""
    parts = []
    eq_pattern = re.compile(r"[-\d\s\(\)\+\-\*\/\.,]+\s*=\s*-?\d+(?:[.,]\d+)?")
    for line in response_vi.split("."):
        line = line.strip()
        if not line or not eq_pattern.search(line):
            continue
        start = re.search(r"\d", line)
        if start:
            parts.append(line[start.start() :].strip())
    return ". ".join(parts) if parts else None


def normalize_response(response: str) -> Optional[str]:
    """Build the final compact target: short computation chain + one answer anchor."""
    gold_str = extract_answer(response, VI_ANCHORS)
    gold_num = parse_number(gold_str)
    if gold_num is None:
        return None

    clean_num = str(int(gold_num)) if gold_num == int(gold_num) else str(gold_num)
    chain = extract_computation_chain(response)
    return f"{chain}\n####đáp án là: {clean_num}" if chain else f"####đáp án là: {clean_num}"


def count_answer_anchors(target: str) -> int:
    return len(re.findall(r"(?i)####\s*đáp\s*án\s*là", target or ""))


def build_prompt(query_vi: str) -> str:
    return PROMPT_TEMPLATE.format(q=query_vi.strip())


def attach_targets(records: list[dict]) -> list[dict]:
    """Attach `_target` to records where a numeric target can be constructed."""
    output = []
    for record in records:
        target = normalize_response(record.get("response_vi", ""))
        if target and count_answer_anchors(target) == 1:
            item = dict(record)
            item["_target"] = target
            output.append(item)
    return output
