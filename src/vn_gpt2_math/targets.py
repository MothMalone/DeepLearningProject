"""Final rewind3-style preprocessing and target construction."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .answers import VI_ANCHORS, extract_answer, parse_number
from .config import PROMPT_TEMPLATE

VIET_DIGIT_MAP: dict[str, int] = {
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


LATEX_FUNCTIONS = (
    "arccos",
    "arcsin",
    "arctan",
    "cos",
    "sin",
    "tan",
    "cot",
    "sec",
    "csc",
    "log",
    "ln",
)


def strip_asy_blocks(text: str) -> str:
    """Remove Asymptote diagram code that burns GPT-2 context and corrupts commas."""
    return re.sub(r"\[asy\].*?(?:\[/asy\]|$)", " ", text, flags=re.IGNORECASE | re.DOTALL)


def normalize_latex_math(text: str) -> str:
    """Conservatively canonicalize common LaTeX math without erasing structure."""
    text = strip_asy_blocks(text)
    text = text.replace("\\$", "").replace("\\%", "%")
    text = re.sub(r"\\angle\s*([A-Z]{1,4})", r"góc \1", text)
    text = re.sub(r"\\triangle\s*([A-Z]{1,4})", r"tam giác \1", text)
    text = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*(\d+(?:[.,]\d+)?)", r"sqrt(\1)", text)

    replacements = {
        "\\times": " * ",
        "\\cdot": " * ",
        "\\div": " / ",
        "\\leq": " <= ",
        "\\le": " <= ",
        "\\geq": " >= ",
        "\\ge": " >= ",
        "\\neq": " != ",
        "\\ne": " != ",
        "\\equiv": " = ",
        "\\pm": " ± ",
        "\\pi": " pi ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    for fn in LATEX_FUNCTIONS:
        text = re.sub(rf"\\{fn}\b", fn, text)

    # Keep the semantic names of frequent variables instead of deleting them.
    greek = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "theta",
        "lambda",
        "mu",
        "phi",
        "omega",
    )
    for name in greek:
        text = re.sub(rf"\\{name}\b", name, text)

    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", text)
    return text


def clean_math_text(text: str) -> str:
    """Light typographical cleanup used by the final notebook."""
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = normalize_latex_math(text)
    text = re.sub(r"\$\$([^$]+)\$\$", r"\1", text)
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
