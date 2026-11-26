"""Static import audit and dependency-declaration detection (never imports or runs team code).

Warn-only: a module that is neither in the image (standard library or release inventory) nor
provided by the repository is reported once, with where it was first imported. Guarded imports
(`try/except ImportError`, `if TYPE_CHECKING`) and relative imports are not reported.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_FILES = 3000
MAX_LISTED = 8
# Directories whose code is not run by a policy (tests, examples, ...). Imports inside them are not
# audited; the same names are the "common names" whose shadow warnings are suppressed (repo.py).
NON_RUNTIME_DIRS = frozenset({"tests", "test", "examples", "docs", "scripts", "cuda", "benchmarks", "notebooks"})
NON_RUNTIME_FILES = frozenset({"conftest.py", "setup.py", "noxfile.py", "fabfile.py"})
_GUARD_EXCEPTIONS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})

_SDPA_HINT = ('use attn_implementation="sdpa" or torch scaled_dot_product_attention (same results); '
              'see "flash-attention to PyTorch SDPA" in contestants/SUBMIT.md')

# Modules teams commonly import that the image does not have, with the supported alternative.
ALTERNATIVES = {
    "flash_attn": _SDPA_HINT,
    "flash_attn_interface": _SDPA_HINT,
    "xformers": _SDPA_HINT,
    "decord": "read video with torchcodec or av",
    "pytorch3d": "use kornia, roma or plain torch",
    "sam2": "load SAM through transformers",
    "segment_anything": "load SAM through transformers",
    "open3d": "use trimesh, numpy or scipy",
    "tensorflow": "convert checkpoints to safetensors or JAX/orbax",
    "tensorrt": "ship the original weights and run them in torch",
    "torch_tensorrt": "ship the original weights and run them in torch",
    "onnx": "only onnxruntime-gpu is installed (no onnx package)",
    "wandb": "there is no network at judging; drop logging",
    "ruckig": "not installed",
    "pybullet": "not installed",
    "vllm": "not installed",
    "torchao": "not installed",
    "robosuite": "not installed",
    "drake": "not installed",
    "pydrake": "not installed",
}

DEPENDENCY_FILES = ("setup.py", "setup.cfg", "environment.yml", "environment.yaml", "pipfile", "pipfile.lock",
                    "poetry.lock", "conda.yaml", "conda.yml")


def _is_guard(handler_type) -> bool:
    if handler_type is None:
        return True
    nodes = handler_type.elts if isinstance(handler_type, ast.Tuple) else [handler_type]
    for node in nodes:
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in _GUARD_EXCEPTIONS:
            return True
    return False


def _is_type_checking(test) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


class _Collector(ast.NodeVisitor):
    def __init__(self):
        self.hits = []  # (top_level_name, lineno)
        self.guard = 0

    def visit_Import(self, node):
        if not self.guard:
            for alias in node.names:
                self.hits.append((alias.name.split(".")[0], node.lineno))

    def visit_ImportFrom(self, node):
        if not self.guard and node.level == 0 and node.module:
            self.hits.append((node.module.split(".")[0], node.lineno))

    def visit_If(self, node):
        if _is_type_checking(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Try(self, node):
        guarded = any(_is_guard(h.type) for h in node.handlers)
        if guarded:
            self.guard += 1
        for part in (node.body, node.handlers):  # a fallback import in the handler is guarded too
            for child in part:
                self.visit(child)
        if guarded:
            self.guard -= 1
        for part in (node.orelse, node.finalbody):
            for child in part:
                self.visit(child)

    visit_TryStar = visit_Try


def imports_of(source: bytes, filename: str = "<src>") -> Optional[list]:
    """[(top_level_name, line)] of unguarded absolute imports, or None when the file does not parse."""
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    collector = _Collector()
    collector.visit(tree)
    return collector.hits


def audited(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in NON_RUNTIME_DIRS for p in parts[:-1]):
        return False
    name = parts[-1]
    return not (name in NON_RUNTIME_FILES or name.startswith("test_") or name.endswith("_test.py"))


def provided_names(py_files: list) -> set:
    """Top-level names importable from the repository root, or from `src/` (the loader adds both)."""
    names = set()
    for rel in py_files:
        parts = rel.split("/")
        for base in (parts, parts[1:] if parts[0] == "src" and len(parts) > 1 else None):
            if not base:
                continue
            names.add(base[0][:-3] if len(base) == 1 and base[0].endswith(".py") else base[0])
    return names


def every_module_name(py_files: list) -> set:
    """Every file stem and directory name in the repository (used when the code edits sys.path)."""
    names = set()
    for rel in py_files:
        parts = rel.split("/")
        names.update(parts[:-1])
        if parts[-1].endswith(".py"):
            names.add(parts[-1][:-3])
    return names


def collect(repo_dir: Path, py_files: list) -> tuple:
    """({rel: [(name, line)] or absent when unparsable}, sees_sys_path) for every readable .py file."""
    per_file: dict = {}
    sees_sys_path = False
    for rel in py_files[:MAX_FILES]:
        path = Path(repo_dir) / rel
        try:
            if path.stat().st_size > MAX_SOURCE_BYTES:
                continue
            source = path.read_bytes()
        except OSError:
            continue
        if b"sys.path" in source:
            sees_sys_path = True
        hits = imports_of(source, rel)
        if hits is not None:
            per_file[rel] = hits
    return per_file, sees_sys_path


def audit(per_file: dict, sees_sys_path: bool, py_files: list, available: frozenset,
          entry_file: Optional[str] = None) -> list:
    """[(module, "path:line")] for modules imported by runtime code but available nowhere, by name.

    ``entry_file`` is the file the entry point runs (program form, or the class form's module).
    It is audited even under tests/, scripts/, examples/ ..., and, like ``python path/to/file.py``,
    the modules next to it are importable.
    """
    provided = provided_names(py_files)
    if entry_file:
        d = entry_file.rsplit("/", 1)[0] if "/" in entry_file else ""
        for rel in py_files:
            head, _, tail = rel.rpartition("/")
            if head == d and tail.endswith(".py"):
                provided.add(tail[:-3])
        for rel in py_files:
            parts = rel.split("/")
            if d and rel.startswith(d + "/") and len(parts) > len(d.split("/")) + 1:
                provided.add(parts[len(d.split("/"))])
    if sees_sys_path:
        provided |= every_module_name(py_files)
    first_seen: dict = {}
    for rel in sorted(per_file):
        if rel != entry_file and not audited(rel):
            continue
        for name, line in per_file[rel]:
            if name in available or name in provided or name in first_seen:
                continue
            first_seen[name] = f"{rel}:{line}"
    return sorted(first_seen.items())


def imported_outside(name: str, per_file: dict) -> bool:
    """True when a .py file that is not the top-level module/package `name` itself imports `name`."""
    for rel, hits in per_file.items():
        if rel == name + ".py" or rel.startswith(name + "/"):
            continue
        if any(hit == name for hit, _ in hits):
            return True
    return False


def format_modules(missing: list) -> tuple:
    """(modules text, alternatives text) for the import_unresolved message."""
    shown = [f"{name} ({where})" for name, where in missing[:MAX_LISTED]]
    if len(missing) > MAX_LISTED:
        shown.append(f"and {len(missing) - MAX_LISTED} more")
    alts = [f"{name}: {ALTERNATIVES[name]}" for name, _ in missing if name in ALTERNATIVES]
    return ", ".join(shown), (" Alternatives: " + "; ".join(alts) + "." if alts else "")


def dependency_files(repo_dir: Path, top_level_files: list) -> list:
    """Top-level files that declare dependencies the image never installs."""
    found = []
    for name in sorted(top_level_files):
        lowered = name.lower()
        if re.match(r"^requirements[A-Za-z0-9_.\-]*\.(txt|in)$", lowered) or lowered in DEPENDENCY_FILES:
            found.append(name)
        elif lowered == "pyproject.toml":
            try:
                text = (Path(repo_dir) / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r"^\s*dependencies\s*=|^\[(tool\.poetry\.dependencies|project\.optional-dependencies|dependency-groups)", text, re.M):
                found.append(name)
    return found
