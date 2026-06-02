"""Small pure helpers for Vietnamese math dataset preprocessing.

These functions intentionally avoid importing the Kaggle notebook because the
notebook has training side effects at module import time.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional


ANSWER_MARKER_PATTERNS = [
    r"Đáp\s*án\s*là\s*[:：]?",
    r"Đáp\s*án\s*[:：]",
    r"Câu\s*trả\s*lời\s*là\s*[:：]?",
    r"Câu\s*trả\s*lời\s*[:：]",
    r"Đáp\s*số\s*là\s*[:：]?",
    r"Đáp\s*số\s*[:：]",
    r"The\s+answer\s+is\s*[:：]?",
    r"####\s*",
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
SAFE_NS = {"sqrt": math.sqrt, "pi": math.pi}


def normalize_space(text: str | None) -> str:
    if text is None:
        return ""
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"{p} must contain a JSON array or JSONL records.")
    return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cleanup_answer_string(text: str | None) -> str:
    ans = normalize_space(text)
    if not ans:
        return ""

    # Cut repeated markers after the first answer content.
    cut_positions: list[int] = []
    for pat in ANSWER_MARKER_PATTERNS:
        m = re.search(pat, ans, flags=re.IGNORECASE)
        if m and m.start() > 0:
            cut_positions.append(m.start())
    if cut_positions:
        ans = ans[: min(cut_positions)].strip()

    ans = ans.split("\n")[0].strip()
    ans = ans.strip(" \t:：")
    ans = ans.strip("\"'“”‘’`")
    ans = ans.rstrip(" .。;；,，")
    ans = ans.strip("\"'“”‘’`")
    return ans


def extract_answer(text: str | None) -> Optional[str]:
    if not text:
        return None
    source = str(text)
    matches: list[tuple[int, int]] = []
    for pat in ANSWER_MARKER_PATTERNS:
        for m in re.finditer(pat, source, flags=re.IGNORECASE):
            matches.append((m.start(), m.end()))
    if matches:
        # Prefer the last marker near the end, which usually contains the final answer.
        start, end = sorted(matches)[-1]
        tail = source[end:].strip()
        return cleanup_answer_string(tail)

    boxes = BOXED_RE.findall(source)
    if boxes:
        return cleanup_answer_string(boxes[-1])
    return None


def _strip_numeric_units(t: str) -> str:
    # Keep a leading scalar/fraction/expression before common unit words.
    m = re.match(
        r"^\s*(-?(?:\d+(?:[.,]\d+)?|\d+\s*/\s*\d+)(?:\s*[%])?)\s+[A-Za-zÀ-ỹđĐ/%$].*$",
        t,
    )
    return m.group(1) if m else t


def parse_number(s: str | None) -> Optional[float]:
    """Best-effort finite scalar parser for final answers."""
    if s is None:
        return None

    t = cleanup_answer_string(str(s))
    if not t:
        return None

    t = t.replace("\u2212", "-").replace("−", "-")
    t = t.strip("$ ").strip("\"'“”‘’`").rstrip(".。;；")

    m = re.match(r"^[A-Za-z_]\w*\s*=\s*(.+)$", t)
    if m:
        t = m.group(1).strip()

    compact0 = re.sub(r"\s+", "", t)
    if (
        (compact0.startswith("(") and compact0.endswith(")") and re.search(r"\d\s*,\s*\d", t))
        or (compact0.startswith("[") and compact0.endswith("]"))
        or re.search(r"\b(?:hoặc|và|and|or)\b", t, flags=re.IGNORECASE)
    ):
        return None

    t = _strip_numeric_units(t)

    for _ in range(3):
        new = re.sub(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", r"(\1)", t)
        if new == t:
            break
        t = new

    t = re.sub(r"\\text\{[^}]*\}", "", t)
    t = re.sub(r"\\mathrm\{[^}]*\}", "", t)
    t = t.replace("$", "")
    for token in ("\\,", "\\!", "\\;", "\\ ", "\\left", "\\right"):
        t = t.replace(token, "")
    for token in ("\\cdot", "\\times", "×"):
        t = t.replace(token, "*")

    t = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", t)
    t = re.sub(r"\\sqrt\s*(\d+(?:[.,]\d+)?)", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi")
    t = t.replace("%", "")

    if re.fullmatch(r"-?\d+,\d+", t):
        t = t.replace(",", ".")
    else:
        t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)

    t = re.sub(r"\s+", "", t)
    if not t or "," in t:
        return None

    t = re.sub(r"(\d)\s*(sqrt|pi|\()", r"\1*\2", t)
    t = re.sub(r"(\))\s*(sqrt|pi|\d)", r"\1*\2", t)
    t = re.sub(r"(pi)\s*(sqrt|pi|\d|\()", r"\1*\2", t)
    leftover = re.sub(r"sqrt|pi|\d|\.|\+|\-|\*|/|\(|\)|\^|e|E", "", t)
    if leftover:
        return None
    t = t.replace("^", "**")

    try:
        val = eval(t, {"__builtins__": {}}, SAFE_NS)  # noqa: S307 - deliberately restricted.
    except Exception:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    val = float(val)
    return val if math.isfinite(val) else None


def numbers_match(a: str | None, b: str | None, tol: float = 1e-9) -> bool:
    ca = cleanup_answer_string(a)
    cb = cleanup_answer_string(b)
    if ca and cb and ca == cb:
        return True
    na = parse_number(ca)
    nb = parse_number(cb)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= tol * max(1.0, abs(nb))


def contains_operator(text: str | Iterable[str]) -> bool:
    joined = "\n".join(text) if not isinstance(text, str) else text
    return bool(re.search(r"(=|\+|\-|\*|/|×|%|\\frac)", joined))


def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first parseable JSON object from a model generation."""
    raw = text.strip()
    if not raw:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fence.group(1)] if fence else []
    candidates.append(raw)

    start_positions = [m.start() for m in re.finditer(r"\{", raw)]
    for start in start_positions:
        depth = 0
        in_str = False
        escape = False
        for pos in range(start, len(raw)):
            ch = raw[pos]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : pos + 1])
                    break

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(obj, dict):
            return obj
    return None
