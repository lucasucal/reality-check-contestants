"""Practice launcher for the pinned policy runtime (contestants/PRACTICE.md).

Reuses, never copies: `rc_checks` (the submission checks), `rc_runner.launch` and
`rc_runner.docker_driver` (the judged `docker run` argv) and `rc_runner.practice` (startup,
warmup and episodes). The sys.path fallback below lets it run from the bundle root without
installing those packages.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "runner", REPO_ROOT / "checks" / "rc_checks"):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.append(str(_extra))
