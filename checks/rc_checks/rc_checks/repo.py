"""Static, filesystem-only checks of a pinned repository clone. Never imports or runs team code."""
from __future__ import annotations

import ast
import json
import os
import sys
import re
import shutil
from pathlib import Path
from typing import Optional

from . import _native, hints, imports
from .findings import CheckResult, Finding, make
from .release import Release, load_release
from .types import Limits

ENTRY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")
KNOWN_MANIFEST_KEYS = frozenset({"contract", "entry_point", "port", "weights"})
RUNTIME_NAMES = frozenset({"rc_sdk", "rc_runtime"})
# The image provides these; a team copy is stripped from the prepared context (with a warning), never run.
RUNTIME_COPY_FILES = frozenset({"rc_sdk.py", "rc_runtime.py"})
# Interpreter start-up hooks stay rejected: they would run before any check could see them.
RESERVED_ROOT_NAMES = frozenset({"sitecustomize", "usercustomize"})
C_EXTS = frozenset({".c", ".cc", ".cpp", ".cxx"})
NATIVE_BUILD_FILES = frozenset({"cmakelists.txt", "cargo.toml"})
MAX_NATIVE_SOURCE_FINDINGS = 10
LEGACY_MANIFEST_MESSAGE = "Check manifest.json is valid JSON with contract v0, an integer port, and weights repo_id and revision."


def _is_bytecode(rel: str) -> bool:
    return rel.endswith((".pyc", ".pyo")) or "__pycache__" in rel.split("/")


def _walk(repo_dir: Path):
    """Yield (rel_posix, abs_path, is_symlink, is_dir) for every entry outside .git, sorted.

    Directories that are symlinks are reported but not descended into. __pycache__ directories
    and top-level rc_sdk/ rc_runtime/ (stripped from the context, so never run) are reported once
    and not descended into.
    """
    for current, dirs, files in os.walk(repo_dir):
        rel_dir = Path(current).relative_to(repo_dir)
        dirs[:] = sorted(d for d in dirs if d != ".git")
        keep = []
        for name in dirs:
            path = Path(current) / name
            rel = (rel_dir / name).as_posix()
            if path.is_symlink():
                yield rel, path, True, True
            else:
                yield rel, path, False, True
                if name != "__pycache__" and not (not rel_dir.parts and name in RUNTIME_NAMES):
                    keep.append(name)
        dirs[:] = keep
        for name in sorted(files):
            if name == ".git":
                continue
            path = Path(current) / name
            yield (rel_dir / name).as_posix(), path, path.is_symlink(), False


def _strict_json(text: str):
    def reject(value):
        raise ValueError(f"{value} is not allowed")
    return json.loads(text, parse_constant=reject)


def _manifest_checks(repo_dir: Path, legacy: bool, weights_required: bool, result: CheckResult):
    """Returns the manifest dict or None."""
    path = repo_dir / "manifest.json"
    if not path.is_file() or path.is_symlink():
        result.findings.append(make("manifest_missing", "manifest.json"))
        return None
    try:
        manifest = _strict_json(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("the top level must be a JSON object")
    except (ValueError, UnicodeError) as exc:
        if legacy:
            result.findings.append(Finding("manifest_invalid", "reject", LEGACY_MANIFEST_MESSAGE, "manifest.json"))
        else:
            result.findings.append(make("manifest_invalid", "manifest.json", detail=str(exc)))
        return None
    if legacy:
        port = manifest.get("port")
        weights = manifest.get("weights")
        bad = (
            manifest.get("contract") != "v0"
            or not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
            or (weights_required and (not isinstance(weights, dict)
                                      or not isinstance(weights.get("repo_id"), str)
                                      or not isinstance(weights.get("revision"), str)))
        )
        if bad:
            result.findings.append(Finding("manifest_invalid", "reject", LEGACY_MANIFEST_MESSAGE, "manifest.json"))
        return manifest
    if manifest.get("contract") != "v0":
        result.findings.append(make("manifest_contract", "manifest.json"))
    if "port" in manifest:
        port = manifest["port"]
        if type(port) is not int or port != 8080:
            result.findings.append(make("manifest_port", "manifest.json"))
    for key in manifest:
        if key not in KNOWN_MANIFEST_KEYS and not key.startswith("_comment"):
            result.findings.append(make("manifest_unknown_key", "manifest.json", key=key))
    return manifest


def _top_level_names(tree: ast.Module) -> set:
    names: set = set()

    def bind_target(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind_target(item)
        elif isinstance(target, ast.Starred):
            bind_target(target.value)

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    bind_target(target)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                bind_target(node.target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                visit(node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                visit(node.body)

    visit(tree.body)
    return names


def _has_star_import(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in tree.body)


MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
PATH_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def classify_entry_point(value) -> tuple:
    """('class'|'module'|'file', None) for a valid entry_point, else ('invalid', reason).

    Same grammar as rc_runtime.loader.parse_entry (the two are tested against each other).
    """
    if not isinstance(value, str) or not value:
        return "invalid", "it must be a non-empty string"
    if ":" in value:
        if ENTRY_RE.match(value):
            return "class", None
        return "invalid", 'a Policy class entry point must look like "package.module:Attr" (letters, digits, underscores, dots; no arguments)'
    if "/" in value or value.endswith(".py") or "\\" in value:
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            return "invalid", "absolute paths are not allowed; use a path relative to the repository root"
        if "\\" in value:
            return "invalid", "use '/' as the path separator"
        parts = value.split("/")
        if any(p in ("", ".", "..") for p in parts) or not all(PATH_PART_RE.match(p) for p in parts):
            return "invalid", "a program path must stay inside the repository (no '..', '.', or empty segments)"
        if not value.endswith(".py"):
            return "invalid", "a program path must end in .py"
        return "file", None
    if MODULE_RE.match(value):
        return "module", None
    return "invalid", "it is neither a dotted module path nor a .py path"


def stripped_by_preparation(rel: str) -> Optional[str]:
    """Why strip_bytecode/strip_runtime_copies would delete the repo-relative path `rel` ("runtime"
    for a top-level rc_sdk/rc_runtime copy, "bytecode" for __pycache__/.pyc/.pyo), else None. The one
    place that knows the strip rules, so entry validation judges the tree the image will see."""
    parts = rel.split("/")
    if parts[0] in RUNTIME_NAMES or (len(parts) == 1 and rel in RUNTIME_COPY_FILES):
        return "runtime"
    if "__pycache__" in parts or rel.endswith((".pyc", ".pyo")):
        return "bytecode"
    return None


def _real_file(repo_dir: Path, rel: str) -> bool:
    """True when rel is a regular file reachable without crossing a symlink, and that preparation
    keeps (a path it strips does not exist for the loader)."""
    if stripped_by_preparation(rel):
        return False
    current = repo_dir
    for part in rel.split("/"):
        current = current / part
        if current.is_symlink():
            return False
    return current.is_file()


IMAGE_PYTHON = (3, 12)  # the pinned image's interpreter; its grammar is the authority


def _syntax_finding(found: str, line: int, detail: str, result: CheckResult) -> None:
    """Hard reject only when this process IS the image's Python: any other interpreter's grammar
    differs (3.11 rejects valid 3.12 code, 3.13 accepts what 3.12 rejects), so a rejection there could
    refuse a valid submission. Elsewhere it is a warning and the image-side load is the authority."""
    code = "entry_syntax_error" if sys.version_info[:2] == IMAGE_PYTHON else "entry_syntax_unverified"
    result.findings.append(make(code, found, line=line, detail=detail))


def _parse_source(repo_dir: Path, found: str, result: CheckResult) -> Optional[ast.Module]:
    kwargs = {"feature_version": IMAGE_PYTHON} if sys.version_info[:2] > IMAGE_PYTHON else {}
    try:
        return ast.parse((repo_dir / found).read_bytes(), filename=found, **kwargs)
    except SyntaxError as exc:
        _syntax_finding(found, exc.lineno or 1, exc.msg, result)
    except (ValueError, RecursionError, MemoryError) as exc:
        _syntax_finding(found, 1, str(exc) or type(exc).__name__, result)
    return None


def _shadow_kind(top: str, release: Release) -> Optional[str]:
    if top in release.stdlib_module_names:
        return "standard library"
    if top in RUNTIME_NAMES:
        return "runtime"
    if top in release.reserved_modules:
        return "installed package"
    return None


def _resolve_module_file(repo_dir: Path, module_path: str, top: str, suffixes: tuple) -> tuple:
    """Find a module's file the way rc_runtime.loader does: /submission first, then /submission/src
    only when `top` is not importable from /submission itself. Returns (found rel or None, candidates
    tried in order). Paths preparation strips never count as present."""
    candidates = [module_path + sfx for sfx in suffixes]
    found = next((c for c in candidates if _real_file(repo_dir, c)), None)
    if found is None and (repo_dir / "src").is_dir() and not (repo_dir / (top + ".py")).exists() \
            and not (repo_dir / top).exists():
        src_candidates = ["src/" + c for c in candidates]
        found = next((c for c in src_candidates if _real_file(repo_dir, c)), None)
        candidates += src_candidates
    return found, candidates


def _program_entry_checks(repo_dir: Path, value: str, kind: str, release: Release,
                          result: CheckResult) -> tuple:
    """Program form. Returns (shadow-exempt top-level name, entry file rel path or None)."""
    if kind == "file":
        if stripped_by_preparation(value) == "runtime":
            result.findings.append(make("entry_shadowed", "manifest.json", name=value.split("/")[0].removesuffix(".py"),
                                        kind="runtime"))
            return value.split("/")[0].removesuffix(".py"), None
        if not _real_file(repo_dir, value):
            result.findings.append(make("entry_program_not_found", value, value=value,
                                        expected=value))
            return value.split("/")[0].removesuffix(".py"), None
        # Run by path, never imported by name: the image cannot shadow it.
        _parse_source(repo_dir, value, result)
        return value.split("/")[0].removesuffix(".py"), value
    top = value.split(".")[0]
    shadow = _shadow_kind(top, release)
    if shadow:
        result.findings.append(make("entry_shadowed", "manifest.json", name=top, kind=shadow))
        return top, None
    module_path = value.replace(".", "/")
    found, candidates = _resolve_module_file(repo_dir, module_path, top, (".py", "/__main__.py"))
    if found is None:
        result.findings.append(make("entry_program_not_found", candidates[0], value=value,
                                    expected=f"{candidates[0]} or {candidates[1]}"))
        return top, None
    _parse_source(repo_dir, found, result)
    return top, found


def _entry_point_checks(repo_dir: Path, manifest: dict, release: Release, result: CheckResult) -> tuple:
    """Returns (entry top-level name or None, entry file rel path or None); the name is returned
    whenever the entry point was syntactically valid, so shadow warnings can exempt it."""
    value = manifest.get("entry_point")
    if value is None or value == "":
        result.findings.append(make("entry_point_missing", "manifest.json"))
        return None, None
    kind, reason = classify_entry_point(value)
    if kind == "invalid":
        result.findings.append(make("entry_point_invalid", "manifest.json", value=value, reason=reason))
        return None, None
    if kind != "class":
        return _program_entry_checks(repo_dir, value, kind, release, result)
    module, _, attr = value.partition(":")
    top = module.split(".")[0]
    # Shadowing by the image comes first: the module on disk would never be loaded.
    shadow = _shadow_kind(top, release)
    if shadow:
        result.findings.append(make("entry_shadowed", "manifest.json", name=top, kind=shadow))
        return top, None
    module_path = module.replace(".", "/")
    found, candidates = _resolve_module_file(repo_dir, module_path, top, (".py", "/__init__.py"))
    if found is None:
        result.findings.append(make("entry_module_not_found", candidates[0], module=module, module_path=module_path))
        return top, None
    tree = _parse_source(repo_dir, found, result)
    if tree is None:
        return top, found
    if attr not in _top_level_names(tree) and not _has_star_import(tree):
        result.findings.append(make("entry_attr_missing", found, attr=attr))
    return top, found


def _fmt_mb(n: int) -> str:
    value = n / (1024 * 1024)
    return f"{value:.0f} MB" if value == int(value) else f"{value:.1f} MB"


def check_repo(repo_dir: Path, *, release: Optional[Release] = None, limits: Limits = Limits(),
               legacy_dockerfile: bool = False, weights_in_manifest_required: bool = True) -> CheckResult:
    repo_dir = Path(repo_dir)
    if release is None:
        release = load_release()
    result = CheckResult()
    manifest = _manifest_checks(repo_dir, legacy_dockerfile, weights_in_manifest_required, result)

    # Dockerfile rule.
    has_dockerfile = (repo_dir / "Dockerfile").is_file()
    if legacy_dockerfile and not has_dockerfile:
        result.findings.append(make("context_no_dockerfile", "Dockerfile"))
    elif not legacy_dockerfile and has_dockerfile:
        result.findings.append(make("dockerfile_ignored", "Dockerfile"))

    # Legacy filesystem rules.
    if (repo_dir / "weights").is_dir():
        result.findings.append(make("weights_in_repo", "weights/"))
    if (repo_dir / ".gitmodules").exists():
        result.findings.append(make("context_submodules", ".gitmodules"))

    total_bytes = total_files = 0
    big_files = []
    bytecode = []
    native_sources = []
    py_files: list = []
    top_files: list = []
    top_level: list = []  # (name, kind) for shadowing, kind in {"file","dir"}
    for rel, path, is_link, is_dir in _walk(repo_dir):
        if "/" not in rel:
            top_level.append((rel, is_dir))
            if not is_dir and not is_link:
                top_files.append(rel)
        if is_link:
            result.findings.append(make("repo_symlink", rel))
            continue
        if is_dir:
            if path.name == "__pycache__":
                bytecode.append(rel + "/")
            continue
        if _is_bytecode(rel):
            if "__pycache__" not in rel.split("/"):
                bytecode.append(rel)
            continue
        if rel in RUNTIME_COPY_FILES:
            continue  # reported by _shadow_checks, stripped before use
        name = path.name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total_files += 1
        total_bytes += size
        if size > limits.max_file_bytes:
            big_files.append((rel, size))
        if 0 < size < 1024:
            try:
                if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                    result.findings.append(make("context_lfs", rel))
                    continue
            except OSError:
                pass
        ext = _native.extension(name)
        if ext in _native.CUDA_SOURCE_EXTS:
            result.findings.append(make("repo_cuda_source", rel))
            continue
        native_ext = _native.native_extension(name)
        reason = f"extension {native_ext}" if native_ext else _native.magic_reason(path, size)
        if not reason and ext in _native.SCANNED_ZIP_EXTS:
            reason = _native.zip_native_reason(path)
        if reason:
            result.findings.append(make("repo_native_binary", rel, reason=reason))
            continue
        if ext in _native.ARCHIVE_EXTS and ext not in _native.ARCHIVE_EXEMPT:
            result.findings.append(make("repo_archive", rel, ext=ext))
            continue
        if ext == ".py":
            py_files.append(rel)
        lowered = name.lower()
        is_native_source = ext in C_EXTS or lowered in NATIVE_BUILD_FILES
        if lowered == "setup.py" and size <= 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                is_native_source = "ext_modules" in text or "Extension(" in text
            except OSError:
                pass
        if is_native_source:
            native_sources.append(rel)

    for rel, size in big_files:
        result.findings.append(make("context_too_heavy", rel, limit=f"{_fmt_mb(limits.max_file_bytes)} per file",
                                    actual=f"{rel} is {_fmt_mb(size)}"))
    if total_bytes > limits.max_bytes:
        result.findings.append(make("context_too_heavy", None, limit=f"{_fmt_mb(limits.max_bytes)} total size",
                                    actual=_fmt_mb(total_bytes)))
    if total_files > limits.max_files:
        result.findings.append(make("context_too_heavy", None, limit=f"{limits.max_files} files",
                                    actual=f"{total_files} files"))
    for rel in native_sources[:MAX_NATIVE_SOURCE_FINDINGS]:
        result.findings.append(make("repo_native_source", rel))
    if bytecode:
        shown = ", ".join(bytecode[:5]) + (", ..." if len(bytecode) > 5 else "")
        result.findings.append(make("repo_bytecode_present", bytecode[0], paths=shown))

    entry_top = entry_file = None
    if not legacy_dockerfile:
        if manifest is not None:
            entry_top, entry_file = _entry_point_checks(repo_dir, manifest, release, result)
        per_file, sees_sys_path = imports.collect(repo_dir, py_files)
        _shadow_checks(top_level, entry_top, release, result, per_file)
        deps = imports.dependency_files(repo_dir, top_files)
        if deps:
            result.findings.append(make("dependencies_ignored", deps[0], files=", ".join(deps)))
        if release.reserved_modules:  # no inventory, no basis for an audit
            available = release.stdlib_module_names | RUNTIME_NAMES | release.reserved_modules | {"__main__"}
            missing = imports.audit(per_file, sees_sys_path, py_files, available, entry_file)
            if missing:
                modules, alternatives = imports.format_modules(missing)
                result.findings.append(make("import_unresolved", None, modules=modules, alternatives=alternatives))
        _hint_checks(repo_dir, py_files, manifest, entry_file, release, result)
    return result


def _hint_checks(repo_dir: Path, py_files: list, manifest: Optional[dict], entry_file: Optional[str],
                 release: Release, result: CheckResult) -> None:
    """Warn-only hints (hints.py) over the files the import audit reads, minus copies the preparation strips."""
    files = [rel for rel in py_files[:imports.MAX_FILES]
             if (rel == entry_file or imports.audited(rel)) and stripped_by_preparation(rel) is None]
    value = (manifest or {}).get("entry_point")
    program_form = isinstance(value, str) and classify_entry_point(value)[0] in ("module", "file")
    found = hints.scan(repo_dir, files, entry_file, program_form, imports.MAX_SOURCE_BYTES)
    if found["hub"]:
        result.findings.append(make("hub_id_offline", found["hub"][0][1].rsplit(":", 1)[0],
                                    ids=hints._listed(found["hub"])))
    if found["flash"]:
        result.findings.append(make("flash_attention_requested", found["flash"][0][1].rsplit(":", 1)[0],
                                    uses=hints._listed(found["flash"])))
    if found["loopback"]:
        host, line = found["loopback"][0]
        result.findings.append(make("program_binds_loopback", entry_file, line=line, host=host))


def _shadow_checks(top_level: list, entry_top: Optional[str], release: Release, result: CheckResult,
                   per_file: Optional[dict] = None) -> None:
    shadowing = release.stdlib_module_names | RUNTIME_NAMES | release.reserved_modules
    for rel, is_dir in top_level:
        if rel.startswith(".") or rel == "__pycache__":
            continue
        if not is_dir and not rel.endswith(".py"):
            continue
        name = rel if is_dir else rel[:-3]
        if name in RESERVED_ROOT_NAMES:
            result.findings.append(make("repo_reserved_name", rel + ("/" if is_dir else "")))
        elif name in RUNTIME_NAMES:
            # An entry point inside it is rejected as entry_shadowed; otherwise the copy is just ignored.
            # The starter ships rc_sdk/ for local runs, so replacing that copy is expected and not worth a warning.
            if name != entry_top and name != "rc_sdk":
                result.findings.append(make("repo_runtime_copy_ignored", rel + ("/" if is_dir else "")))
        elif name in shadowing and name != entry_top:
            # tests/, examples/, scripts/ ... collide with stray image modules all the time and are
            # never run by a policy: warn only when other code in the repository imports the name.
            if name in imports.NON_RUNTIME_DIRS and not imports.imported_outside(name, per_file or {}):
                continue
            result.findings.append(make("repo_module_shadowed", rel + ("/" if is_dir else ""), name=name))


def strip_bytecode(repo_dir: Path) -> list:
    """Delete __pycache__ dirs and *.pyc/*.pyo; return removed relative paths."""
    repo_dir = Path(repo_dir)
    removed = []
    for current, dirs, files in os.walk(repo_dir):
        rel_dir = Path(current).relative_to(repo_dir)
        if ".git" in rel_dir.parts:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in list(dirs):
            path = Path(current) / name
            if name == "__pycache__" and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
                removed.append((rel_dir / name).as_posix() + "/")
                dirs.remove(name)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                try:
                    (Path(current) / name).unlink()
                    removed.append((rel_dir / name).as_posix())
                except OSError:
                    pass
    return sorted(removed)


def strip_runtime_copies(repo_dir: Path) -> list:
    """Delete a team's top-level rc_sdk/ rc_runtime/ (or rc_sdk.py/rc_runtime.py); the image provides
    them. Return removed relative paths. Every place that prepares /submission calls this, so the
    image never sees a team copy. Symlinks are unlinked, not followed."""
    repo_dir = Path(repo_dir)
    removed = []
    for name in sorted(RUNTIME_NAMES):
        for rel, is_dir in ((name, True), (name + ".py", False)):
            path = repo_dir / rel
            if path.is_symlink() or (path.is_file() and not is_dir):
                path.unlink()
            elif is_dir and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                continue
            removed.append(rel + ("/" if is_dir else ""))
    return sorted(removed)
