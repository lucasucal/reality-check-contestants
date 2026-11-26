"""release.json loading (contract section 2)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ReleaseError(Exception):
    pass


@dataclass(frozen=True)
class Release:
    raw: dict
    release_id: str
    python_version: str
    reserved_modules: frozenset
    stdlib_module_names: frozenset
    numeric_formats: tuple
    base_cache_version: Optional[str]
    docs_url: str


def _interpreter_stdlib_names() -> frozenset:
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    import pkgutil
    import sysconfig
    found = set(sys.builtin_module_names)
    stdlib = sysconfig.get_paths().get("stdlib")
    if stdlib:
        found.update(m.name for m in pkgutil.iter_modules([stdlib]))
    return frozenset(found)


def _str_list(raw: dict, key: str) -> list:
    value = raw.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ReleaseError(f"release.json {key!r} must be a list of strings")
    return value


def parse_release(raw: object) -> Release:
    if not isinstance(raw, dict):
        raise ReleaseError("release.json must be a JSON object")
    version = raw.get("schema_version", 1)
    if version != 1:
        raise ReleaseError(f"unsupported release.json schema_version {version!r}")
    release_id = raw.get("release_id", "")
    if not isinstance(release_id, str):
        raise ReleaseError("release.json release_id must be a string")
    python = raw.get("python") or {}
    if not isinstance(python, dict):
        raise ReleaseError("release.json python must be an object")
    formats = raw.get("numeric_formats", [])
    if formats is None:
        formats = []
    if not isinstance(formats, list) or not all(isinstance(x, dict) for x in formats):
        raise ReleaseError("release.json numeric_formats must be a list of objects")
    for row in formats:
        codes = row.get("dtype_codes", [])
        if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
            raise ReleaseError("numeric_formats dtype_codes must be a list of strings")
    stdlib = raw.get("stdlib_module_names")
    stdlib_names = frozenset(_str_list(raw, "stdlib_module_names")) if stdlib else _interpreter_stdlib_names()
    base_cache = raw.get("base_cache") or {}
    if not isinstance(base_cache, dict):
        raise ReleaseError("release.json base_cache must be an object")
    cache_version = base_cache.get("version")
    docs_url = raw.get("docs_url", "/docs")
    if not isinstance(docs_url, str):
        raise ReleaseError("release.json docs_url must be a string")
    return Release(
        raw=raw,
        release_id=release_id,
        python_version=str(python.get("version", "")),
        reserved_modules=frozenset(_str_list(raw, "reserved_modules")),
        stdlib_module_names=stdlib_names,
        numeric_formats=tuple(formats),
        base_cache_version=cache_version if isinstance(cache_version, str) else None,
        docs_url=docs_url.rstrip("/"),
    )


FALLBACK_PATH = Path(__file__).with_name("_release_fallback.json")


def default_release_path() -> Path:
    """$RC_RELEASE_JSON, else <repo>/runtime/release.json, else the packaged fallback (which names no image)."""
    env = os.environ.get("RC_RELEASE_JSON")
    if env:
        return Path(env)
    candidate = Path(__file__).resolve().parents[3] / "runtime" / "release.json"
    return candidate if candidate.is_file() else FALLBACK_PATH


def load_release(path: Optional[Path] = None) -> Release:
    """path None: $RC_RELEASE_JSON, else <repo>/runtime/release.json, else the packaged fallback."""
    if path is None:
        path = default_release_path()
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"cannot read release file {path}: {exc}") from exc
    return parse_release(raw)
