"""Shared static checks for Reality Check submissions (stdlib only)."""
from .dtypes import check_dtypes
from .findings import CheckResult, Finding, Severity, render
from .hashing import snapshot_manifest, tree_sha256, verify_snapshot_manifest
from .release import Release, ReleaseError, load_release
from .repo import check_repo, strip_bytecode, strip_runtime_copies
from .snapshot import check_snapshot
from .types import Limits

__all__ = [
    "CheckResult", "Finding", "Limits", "Release", "ReleaseError", "Severity",
    "check_dtypes", "check_repo", "check_snapshot", "load_release", "render",
    "snapshot_manifest", "strip_bytecode", "strip_runtime_copies", "tree_sha256", "verify_snapshot_manifest",
]
