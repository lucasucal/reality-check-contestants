"""Run the shared rc_checks over the exported repo and the HF snapshot. Messages come from
rc_checks' own renderer, so they are byte-identical to what the backend shows on a receipt."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import rc_checks
from rc_checks.dtypes import read_safetensors_dtypes


def format_finding(f: "rc_checks.Finding") -> str:
    """Same line shape as `python -m rc_checks`."""
    return f"{f.severity} {f.code} {f.path or '-'}: {f.message}"


@dataclass
class CheckOutcome:
    repo: "rc_checks.CheckResult"
    snapshot: "rc_checks.CheckResult"
    dtypes: "rc_checks.CheckResult"
    dtype_codes: frozenset = field(default_factory=frozenset)
    unreadable: tuple = ()

    @property
    def ok(self) -> bool:
        return self.repo.ok and self.snapshot.ok

    def lines(self) -> list:
        out = []
        for title, result in (("repository", self.repo), ("model snapshot", self.snapshot), ("numeric formats", self.dtypes)):
            findings = result.to_dict()["findings"]
            out.append(f"Checks: {title}: " + ("no findings" if not findings else f"{len(findings)} finding(s)"))
            out.extend("  " + format_finding(rc_checks.Finding(f["code"], f["severity"], f["message"], f["path"], f.get("line")))
                       for f in findings)
        return out


def snapshot_dtype_codes(snapshot_dir: Path) -> tuple:
    """(set of safetensors dtype codes found, unreadable files). Headers only."""
    codes: set = set()
    bad = []
    for current, dirs, files in os.walk(snapshot_dir):
        if Path(current) == Path(snapshot_dir) and ".cache" in dirs:
            dirs.remove(".cache")
        for name in sorted(files):
            path = Path(current) / name
            if name.endswith(".safetensors") and not path.is_symlink():
                try:
                    codes |= read_safetensors_dtypes(path)
                except (OSError, ValueError):
                    bad.append(path.relative_to(snapshot_dir).as_posix())
    return frozenset(codes), tuple(bad)


def run_checks(repo_tree: Path, snapshot_dir: Path, release: "rc_checks.Release", *,
               limits: Optional["rc_checks.Limits"] = None) -> CheckOutcome:
    """Same calls the backend makes (pinned_image mode): check_repo, check_snapshot, check_dtypes."""
    repo = rc_checks.check_repo(repo_tree, release=release, limits=limits or rc_checks.Limits())
    rc_checks.strip_bytecode(repo_tree)
    rc_checks.strip_runtime_copies(repo_tree)
    snap = rc_checks.check_snapshot(snapshot_dir, release=release)
    dtypes = rc_checks.check_dtypes(snapshot_dir, release=release)
    codes, bad = snapshot_dtype_codes(snapshot_dir)
    return CheckOutcome(repo, snap, dtypes, codes, bad)
