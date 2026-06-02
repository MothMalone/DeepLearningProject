"""Answer extraction, numeric parsing, output cleanup, and scoring."""

from __future__ import annotations

import math
import re
from typing import Optional

VI_ANCHORS = [
    r"(?i)(?:đáp án|câu trả lời)\s*là\s*[:：]?\s*(\\frac\{[^{}]+\}\{[^{}]+\})",
    r"(?i)(?:đáp án|câu trả lời)\s*là\s*[:：]?\s*(\\sqrt\{[^{}]+\})",
    r"(?i)####\s*đáp\s*án\s*là\s*[:：]?\s*(\\frac\{[^{}]+\}\{[^{}]+\})",
    r"(?i)####\s*đáp\s*án\s*là\s*[:：]?\s*(\\sqrt\{[^{}]+\})",
    r"(?i)(?:đáp án|câu trả lời)\s*là\s*[:：]?\s*(-?\d+(?:[.,]\d+)?)",
    r"(?i)####đáp án là:\s*(.*?)(?=$|\n)",
    r"(?i)(?:đáp án|câu trả lời)\s*là\s*[:：]?\s*(.*?)(?=$|\n)",
    r"####\s*(.*?)(?=$|\n)",
    r"####đáp án là:\s*",
    r"####\s*",
    r"Đáp án là\s*[:：]?",
    r"Câu trả lời là\s*[:：]?",
]

EN_ANCHORS = [
    r"(?i)the answer is\s*[:：]?\s*(-?\d+(?:[.,]\d+)?)",
    r"The answer is\s*[:：]?",
    r"####",
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
SAFE_NS = {"sqrt": math.sqrt, "pi": math.pi}
LATEX_KEEP = re.compile(
    r"(?:-?\d+\\sqrt\{[^{}]+\}|\\sqrt\{[^{}]+\}|-?\d+\\frac\{[^{}]+\}\{[^{}]+\}|\\frac\{[^{}]+\}\{[^{}]+\})"
)


def clean_model_output(text: str) -> str:
    """Strip trailing garbage after the final answer anchor without changing valid LaTeX answers."""
    if not text:
        return text

    m = re.search(r"(?i)(####\s*đáp\s*án\s*là\s*[:：]?\s*)", text)
    if not m:
        return text

    prefix = text[: m.end()]
    tail = text[m.end() :].lstrip()

    latex_m = LATEX_KEEP.match(tail)
    if latex_m:
        return prefix + latex_m.group(0)

    num_m = re.match(r"(-?\d+(?:[.,]\d+)?)", tail)
    if num_m:
        return prefix + num_m.group(1)

    return prefix + tail.split()[0] if tail.split() else prefix + tail


def extract_answer(text: Optional[str], anchors: list[str] | None = None) -> Optional[str]:
    if not text:
        return None

    for anchor in anchors or VI_ANCHORS:
        match = re.search(anchor, text)
        if match:
            if match.lastindex:
                return match.group(1).strip().rstrip(".。、,")
            tail = text[match.end() :].strip().split("\n")[0]
            return tail.strip().rstrip(".。、,")

    boxes = BOXED_RE.findall(text)
    return boxes[-1].strip() if boxes else None


def extract_gold(record: dict) -> Optional[str]:
    return extract_answer(record.get("response_vi"), VI_ANCHORS)


def extract_pred(record: dict) -> Optional[str]:
    text = record.get("model_output", "")
    answer = extract_answer(clean_model_output(text), VI_ANCHORS + EN_ANCHORS)
    if answer is not None:
        latex_m = LATEX_KEEP.match(answer)
        if latex_m:
            return latex_m.group(0)
        num_m = re.search(r"-?\d+(?:[.,]\d+)?", answer)
        return num_m.group(0) if num_m else None
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    return nums[-1] if nums else None


def parse_number(value: Optional[str]) -> Optional[float]:
    """Best-effort safe parser for scalar numeric answers."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    if re.fullmatch(r"-?\d+,\d+", text):
        text = text.replace(",", ".")

    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        try:
            parsed = float(text)
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None

    assign = re.match(r"^[A-Za-z_]\w*\s*=\s*(.+)$", text)
    if assign:
        text = assign.group(1).strip()

    if text.startswith("(") and text.endswith(")") and re.search(r"\d\s*,\s*\d", text):
        return None
    if text.startswith("[") and text.endswith("]"):
        return None

    for _ in range(3):
        new = re.sub(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r"(\1)", text)
        if new == text:
            break
        text = new

    text = re.sub(r"\\text\{[^}]*\}", "", text)
    text = re.sub(r"\\mathrm\{[^}]*\}", "", text)
    text = text.replace("$", "")
    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right"):
        text = text.replace(token, "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi")
    text = re.sub(r"([\d.])\s*(sqrt|pi|\()", r"\1*\2", text)
    text = re.sub(r"(\))\s*(sqrt|pi|[\d.])", r"\1*\2", text)
    text = re.sub(r"(pi)\s*(sqrt|pi|[\d.]|\()", r"\1*\2", text)
    text = re.sub(r"\)\s*\(", r")*(", text)

    if text.count(",") == 1 and "." not in text and re.search(r"\d,\d", text):
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    elif "," in text:
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    text = re.sub(r"\s+", "", text)
    if not text or "," in text:
        return None

    leftover = re.sub(r"sqrt|pi|\d|\.|\+|\-|\*|/|\(|\)|\^|e|E", "", text)
    if leftover:
        return None

    try:
        parsed = eval(text.replace("^", "**"), {"__builtins__": {}}, SAFE_NS)
    except Exception:
        return None

    if isinstance(parsed, bool):
        return None
    if isinstance(parsed, (int, float)):
        parsed = float(parsed)
        return parsed if math.isfinite(parsed) else None
    return None


def rel_error(pred: Optional[float], gold: Optional[float]) -> Optional[float]:
    if pred is None or gold is None:
        return None
    return abs(pred - gold) / max(1.0, abs(gold))


def score_one(error: Optional[float], extractable: bool) -> int:
    if not extractable or error is None:
        return 0
    if error <= 0.01:
        return 10
    if error <= 0.10:
        return 5
    if error <= 0.50:
        return 1
    return 0
