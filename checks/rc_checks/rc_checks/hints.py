"""Static hints for code that will fail inside the judged sandbox (warn only, never a rejection).

Three patterns found by contestant simulations (CS-B-6, 7, 8), read from the AST of the files the import
audit reads (runtime code plus the entry file); team code is never imported or run:

* a Hugging Face hub id string passed to from_pretrained / hf_hub_download / snapshot_download, which cannot
  resolve offline unless the model is on the curated base list (mounted at /hf-base);
* attn_implementation "flash_attention_2" (flash-attn is not in the image);
* a program-form entry file binding its server to 127.0.0.1 / localhost (never reachable by the runner).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Optional

LOAD_FUNCS = frozenset({"from_pretrained", "hf_hub_download", "snapshot_download"})
ID_KWARGS = ("pretrained_model_name_or_path", "repo_id", "pretrained_model_name", "model_id")
HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
FLASH_STRINGS = frozenset({"flash_attention_2", "flash_attention_3"})
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
HOST_KWARGS = frozenset({"host", "hostname", "bind", "server_name"})
MAX_LISTED = 5


def _func_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _str(node: Optional[ast.AST]) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _module_constants(tree: ast.Module) -> dict:
    """NAME = "literal" at module top level (single target), the usual MODEL_ID pattern."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _str(node.value)
            if value is not None:
                out[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _str(node.value)
            if value is not None:
                out[node.target.id] = value
    return out


def _parse(repo_dir: Path, rel: str, max_bytes: int) -> Optional[ast.Module]:
    path = Path(repo_dir) / rel
    try:
        if path.stat().st_size > max_bytes:
            return None
        return ast.parse(path.read_bytes(), filename=rel)
    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _looks_like_local(repo_dir: Path, rel: str, value: str) -> bool:
    if value.startswith(("/", ".", "~")) or not HUB_ID_RE.match(value):
        return True
    here = (Path(repo_dir) / rel).parent
    return (Path(repo_dir) / value).exists() or (here / value).exists()


def hub_ids(repo_dir: Path, tree: ast.Module, rel: str, curated: set) -> list:
    """[(hub_id, "rel:line")] for literal hub ids passed to the hub loaders."""
    consts = _module_constants(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _func_name(node.func) not in LOAD_FUNCS:
            continue
        arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ID_KWARGS), None)
        value = _str(arg)
        if value is None and isinstance(arg, ast.Name):
            value = consts.get(arg.id)
        if value is None or value in curated or _looks_like_local(repo_dir, rel, value):
            continue
        found.append((value, f"{rel}:{node.lineno}"))
    return found


def flash_attention(tree: ast.Module, rel: str) -> list:
    """[(string, "rel:line")] for flash-attention implementation names anywhere in the file."""
    return [(n.value, f"{rel}:{n.lineno}") for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in FLASH_STRINGS]


def loopback_binds(tree: ast.Module) -> list:
    """[(host, line)] where a call gets host=<loopback> or a (<loopback>, port) address tuple."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            host = _str(kw.value)
            if kw.arg in HOST_KWARGS and host in LOOPBACK:
                out.append((host, kw.value.lineno))
        for arg in node.args:
            if isinstance(arg, ast.Tuple) and len(arg.elts) == 2 and _str(arg.elts[0]) in LOOPBACK:
                out.append((_str(arg.elts[0]), arg.lineno))
    return sorted(set(out), key=lambda x: x[1])


def _listed(items: Iterable[tuple]) -> str:
    items = list(items)
    shown = [f"{value!r} ({where})" for value, where in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        shown.append(f"and {len(items) - MAX_LISTED} more")
    return ", ".join(shown)


def scan(repo_dir: Path, files: list, entry_file: Optional[str], program_form: bool, curated: set,
         max_bytes: int) -> dict:
    """{"hub": [(id, where)], "flash": [(str, where)], "loopback": [(host, line)]} over `files`."""
    hub, flash, loop = [], [], []
    seen_ids = set()
    for rel in files:
        tree = _parse(repo_dir, rel, max_bytes)
        if tree is None:
            continue
        for value, where in hub_ids(repo_dir, tree, rel, curated):
            if value not in seen_ids:
                seen_ids.add(value)
                hub.append((value, where))
        flash.extend(flash_attention(tree, rel))
        if program_form and rel == entry_file:
            loop = loopback_binds(tree)
    return {"hub": hub, "flash": flash, "loopback": loop}


def hub_list_note(curated: set) -> str:
    if curated:
        return ("only the organizer-curated base models resolve by hub id (mounted read-only at /hf-base), and these "
                "ids are not on that list.")
    return ("only the organizer-curated base models resolve by hub id (mounted read-only at /hf-base), and that list "
            "is not published yet, so plan for your load to read only /weights or /hf-base.")
