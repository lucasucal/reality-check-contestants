"""Deterministic tree and snapshot hashing shared by backend and runner."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Sequence

_BYTECODE = (".pyc", ".pyo")


def _sha256_file(path: Path) -> tuple:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _walk_files(root: Path, skip_dirs: Sequence[str] = (), skip_top: Sequence[str] = ()) -> Iterator[tuple]:
    """Yield (posix_rel, abs) for regular non-symlink files; symlinks are yielded as ('rel', None)."""
    for current, dirs, files in os.walk(root):
        rel_dir = Path(current).relative_to(root)
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not (rel_dir == Path(".") and d in skip_top))
        for name in sorted(files):
            yield (rel_dir / name).as_posix(), Path(current) / name
        # symlinked directories are listed in dirs but never descended into by os.walk
        for name in list(dirs):
            if os.path.islink(Path(current) / name):
                dirs.remove(name)
                yield (rel_dir / name).as_posix(), None


def tree_sha256(root: Path) -> str:
    root = Path(root)
    entries = []
    for rel, path in _walk_files(root, skip_dirs=(".git", "__pycache__")):
        if path is None or path.is_symlink() or rel.endswith(_BYTECODE):
            continue
        entries.append((rel, path))
    combined = hashlib.sha256()
    for rel, path in sorted(entries):
        hexdigest, size = _sha256_file(path)
        combined.update(f"{rel}\0{hexdigest}\0{size}\n".encode())
    return "sha256:" + combined.hexdigest()


def _canonical(files: list) -> bytes:
    return json.dumps(files, sort_keys=True, separators=(",", ":")).encode()


def snapshot_manifest(root: Path, *, skip: Sequence[str] = (".cache",)) -> dict:
    root = Path(root)
    files = []
    total = 0
    for rel, path in _walk_files(root, skip_top=skip):
        if path is None or path.is_symlink():
            continue
        hexdigest, size = _sha256_file(path)
        files.append({"path": rel, "size": size, "sha256": hexdigest})
        total += size
    files.sort(key=lambda f: f["path"])
    return {
        "version": 1,
        "files": files,
        "total_bytes": total,
        "manifest_sha256": "sha256:" + hashlib.sha256(_canonical(files)).hexdigest(),
    }


def verify_snapshot_manifest(root: Path, manifest: dict, *, deep: bool = False,
                             skip: Sequence[str] = (".cache",)) -> list:
    root = Path(root)
    problems = []
    expected = {f["path"]: f for f in manifest.get("files", [])}
    seen = set()
    for rel, path in _walk_files(root, skip_top=skip):
        seen.add(rel)
        if path is None or path.is_symlink():
            problems.append(f"{rel}: symlink")
        elif rel not in expected:
            problems.append(f"{rel}: unexpected file")
    for rel, entry in sorted(expected.items()):
        path = root / rel
        if rel not in seen or not path.is_file() or path.is_symlink():
            if rel not in seen:
                problems.append(f"{rel}: missing")
            continue
        if path.stat().st_size != entry["size"]:
            problems.append(f"{rel}: size mismatch")
        elif deep and _sha256_file(path)[0] != entry["sha256"]:
            problems.append(f"{rel}: sha256 mismatch")
    return problems
