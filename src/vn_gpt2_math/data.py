"""Data and artifact I/O helpers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def first_existing(*paths: str | Path) -> Path:
    """Return the first path that exists, or raise a clear error."""
    for path in map(Path, paths):
        if path.exists():
            return path
    raise FileNotFoundError("No candidate path exists: " + " | ".join(map(str, paths)))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL file and normalize string fields to NFC."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        records = json.load(f) if head == "[" else [json.loads(line) for line in f if line.strip()]

    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, str):
                record[key] = unicodedata.normalize("NFC", value)
    return records


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(dir_path: str | Path, suffixes: tuple[str, ...] = (".bin", ".safetensors", ".json", ".txt", ".model")) -> str:
    dir_path = Path(dir_path)
    h = hashlib.sha256()
    for path in sorted(x for x in dir_path.rglob("*") if x.is_file() and x.suffix in suffixes):
        h.update(path.relative_to(dir_path).as_posix().encode() + b"\0")
        h.update(sha256_file(path).encode() + b"\0")
    return h.hexdigest()
