"""safetensors header dtype check. Warnings only; never reads tensor data."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from . import _native
from .findings import CheckResult, make
from .release import Release

HEADER_CAP = 100 * 1024 * 1024
BASELINE = frozenset({"F32", "BF16", "F16"})
NEUTRAL = BASELINE | frozenset({"F64", "I64", "I32", "I16", "U8", "U16", "U32", "U64", "BOOL", "C64"})
UNCHECKED_EXTS = (".bin", ".pt", ".pth", ".ckpt", ".npz", ".npy", ".gguf")


def read_safetensors_dtypes(path: Path) -> set:
    """Return the set of dtype codes in a safetensors header. Raises ValueError if unreadable."""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("file too short")
        (length,) = struct.unpack("<Q", raw)
        if length > HEADER_CAP or length > size - 8:
            raise ValueError("header length out of range")
        body = handle.read(length)
    try:
        header = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("header is not JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    found = set()
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or not isinstance(value.get("dtype"), str):
            raise ValueError("tensor entry without dtype")
        found.add(value["dtype"])
    return found


def _lookup(release: Release) -> dict:
    table = {}
    for row in release.numeric_formats:
        for code in row.get("dtype_codes", []):
            table[code] = row
    return table


def check_dtypes(snapshot_dir: Path, *, release: Release) -> CheckResult:
    root = Path(snapshot_dir)
    result = CheckResult()
    table = _lookup(release)
    unchecked: dict = {}
    docs = release.docs_url
    for current, dirs, files in os.walk(root):
        rel_dir = Path(current).relative_to(root)
        if rel_dir == Path(".") and ".cache" in dirs:
            dirs.remove(".cache")
        dirs.sort()
        for name in sorted(files):
            path = Path(current) / name
            if path.is_symlink():
                continue
            rel = (rel_dir / name).as_posix()
            ext = _native.extension(name)
            if ext == ".safetensors":
                try:
                    codes = read_safetensors_dtypes(path)
                except (OSError, ValueError):
                    result.findings.append(make("dtype_header_unreadable", rel, docs=docs))
                    continue
                flagged = []
                for code in sorted(codes - NEUTRAL):
                    row = table.get(code)
                    if row is not None and row.get("baseline") is True:
                        continue
                    display = row.get("display") if row else None
                    flagged.append(f"{display} ({code})" if display and display != code else code)
                if flagged:
                    result.findings.append(make("dtype_non_baseline", rel, formats=", ".join(flagged), docs=docs))
            elif ext in UNCHECKED_EXTS:
                unchecked.setdefault(ext, []).append(rel)
    for ext in UNCHECKED_EXTS:
        paths = unchecked.get(ext)
        if paths:
            result.findings.append(make("dtype_unchecked_format", paths[0], n=len(paths), ext=ext,
                                        example=paths[0], docs=docs))
    return result
