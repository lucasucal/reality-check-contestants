from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 200 * 1024 * 1024
    max_files: int = 2000
    max_file_bytes: int = 20 * 1024 * 1024
