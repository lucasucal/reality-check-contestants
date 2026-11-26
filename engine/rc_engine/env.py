"""Thin internal env interface; one adapter per engine (docs/decisions.md D7)."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .schema import JointDelta, RawObs, TaskSpec


@dataclass
class StepResult:
    obs: RawObs
    done: bool
    success: bool
    # Episode result. Defaults keep tasks that are still boolean-only constructable.
    # The policy observation is not this object.
    score: float = 0.0
    doneness: float = 0.0
    speed: float = 0.0
    energy: float = 0.0
    # Finish quality Q (score.py) and the measurement that set it. Left None, a task without pose
    # tolerances (bench, cups, placeholder, dummy) reports Q = 1.0 on success and 0.0 otherwise.
    quality: float | None = None
    quality_limit: str | None = None
    limits: dict | None = None

    def __post_init__(self) -> None:
        if self.quality is None:
            self.quality = 1.0 if self.success else 0.0


class Env(Protocol):
    spec: TaskSpec
    step_count: int

    def reset(self, seed: int) -> RawObs: ...

    def step(self, action: JointDelta) -> StepResult: ...

    def state_hash(self) -> str: ...


def hash_state(step: int, *arrays: np.ndarray) -> str:
    """sha256 over step (uint64 LE) then each array as little-endian float64."""
    h = hashlib.sha256(struct.pack("<Q", step))
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return h.hexdigest()
