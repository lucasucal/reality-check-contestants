"""Observation/action schema — the single source of truth (docs/platform.md §6.3).

v2: UR12e (6 arm joints + gripper). Changing the embodiment changes this file and the
task modules only. /tasks exports the specs built here, and parse_action enforces the
same constants. There is no ee_delta.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

SCHEMA_VERSION = 2
NUM_JOINTS = 6  # UR12e arm: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
# rad per 20 Hz control step, per joint: the UR12e's maximum joint speeds (120 °/s base and shoulder, 180 °/s elbow
# and wrists) times 0.05 s. arm_limits.py adds the 1.5 m/s TCP and elbow speed limits.
MAX_JOINT_DELTA: tuple[float, ...] = (0.10472, 0.10472, 0.15708, 0.15708, 0.15708, 0.15708)
TCP_SPEED_LIMIT = 1.5  # m/s
ELBOW_SPEED_LIMIT = 1.5  # m/s
MAX_CHUNK = 16
GOAL_WIDTH = 640
GOAL_HEIGHT = 480


class ProtocolViolation(ValueError):
    """An action the engine refuses; surfaced as 400 policy_protocol_violation."""


@dataclass(frozen=True)
class JointDelta:
    values: tuple[float, ...]
    gripper: float  # 0 closed .. 1 open


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    description: str
    episode_length: int
    control_hz: int
    cameras: tuple[CameraSpec, ...]
    family: str = ""
    visibility: str = "public"  # "hidden" tasks come from a mounted task pack
    instruction: str = ""  # sentence sent on every /act; empty falls back to description
    wire_id: str = ""  # hidden contest tasks: "1" to "4". Empty keeps the legacy wire id "hidden".
    # Goal-picture key of the episode's scene when a task has per-seed variants; empty means task_id.
    # The task list always shows the task_id picture (the first variant's).
    goal_key: str = ""
    # Public tasks: every phrasing an episode may send (the pool). `instruction` is the current episode's.
    instructions: tuple[str, ...] = ()

    def goal_id(self) -> str:
        return self.goal_key or self.task_id

    def policy_task_id(self) -> str:
        """task_id on the wire. Hidden contest tasks are "1" to "4"."""
        if self.visibility != "hidden":
            return self.task_id
        return self.wire_id or "hidden"

    def sentence(self) -> str:
        return self.instruction or self.description

    def to_json(self, jpeg_quality: int = 90) -> dict[str, Any]:
        body: dict[str, Any] = {
            "task_id": self.task_id,
            "description": self.description,
            "episode_length": self.episode_length,
            "control_hz": self.control_hz,
            "family": self.family,
            "visibility": self.visibility,
            "action_spec": action_spec(),
            "obs_spec": obs_spec(self),
        }
        # Public task list carries the sentence and the goal. Hidden tasks do not:
        # those show up in /act when the episode starts.
        if self.visibility == "public":
            # The first phrasing and the canonical goal; each episode draws its own phrasing and scene.
            body["instruction"] = self.instructions[0] if self.instructions else self.sentence()
            if self.instructions:
                body["instructions"] = list(self.instructions)
            body["goal"] = goal_jpeg(self.task_id, jpeg_quality)
        return body


@dataclass
class RawObs:
    """Unencoded observation; frames stay arrays until encode_obs (binary mode later)."""

    images: dict[str, np.ndarray]  # name -> uint8 HxWx3
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    gripper: float
    ee_pos: np.ndarray
    ee_quat: np.ndarray  # wxyz, world frame


def action_spec() -> dict[str, Any]:
    return {
        "types": ["joint_delta", "action_chunk"],
        "joint_delta": {
            "values": {"shape": [NUM_JOINTS], "unit": "rad", "min": [-x for x in MAX_JOINT_DELTA],
                       "max": list(MAX_JOINT_DELTA)},
            "gripper": {"min": 0.0, "max": 1.0, "meaning": "0 closed, 1 open"},
            "speed_limit": {"tcp_m_s": TCP_SPEED_LIMIT, "elbow_m_s": ELBOW_SPEED_LIMIT,
                            "rule": "a faster step is scaled down as a whole, direction kept"},
        },
        "action_chunk": {
            "actions": {"min_len": 1, "max_len": MAX_CHUNK, "item": "joint_delta"},
            "chunk_len": "optional; must equal len(actions)",
        },
        "ignored_keys": "keys starting with '_'",
    }


def obs_spec(task: TaskSpec) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "instruction": "string",
        "goal": {"encoding": "jpeg", "width": GOAL_WIDTH, "height": GOAL_HEIGHT},
        "images": {c.name: {"encoding": "jpeg", "width": c.width, "height": c.height} for c in task.cameras},
        "proprio": {
            "joint_pos": [NUM_JOINTS],
            "joint_vel": [NUM_JOINTS],
            "gripper": "0 closed .. 1 open",
            "ee_pos": [3],
            "ee_quat": [4],
        },
        "proprio_units": {"joint_pos": "rad", "joint_vel": "rad/s", "ee_pos": "m",
                          "ee_quat": "unit quaternion wxyz, world frame"},
        "image_color": "RGB after JPEG decoding",
    }


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolViolation(f"{where} must be a number, got {type(value).__name__}")
    try:
        x = float(value)
    except OverflowError:
        raise ProtocolViolation(f"{where} is too large") from None
    if not math.isfinite(x):
        raise ProtocolViolation(f"{where}={value} is not finite")
    return x


def _check_keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(k for k in raw if k not in allowed and not k.startswith("_"))
    if unknown:
        raise ProtocolViolation(f"{where} has unknown keys: {', '.join(unknown)}")


def _parse_joint_delta(raw: Any, where: str) -> JointDelta:
    if not isinstance(raw, dict):
        raise ProtocolViolation(f"{where} must be an object")
    if raw.get("type") != "joint_delta":
        raise ProtocolViolation(f"{where}.type must be 'joint_delta', got {raw.get('type')!r}")
    _check_keys(raw, {"type", "values", "gripper"}, where)
    values = raw.get("values")
    if not isinstance(values, list) or len(values) != NUM_JOINTS:
        got = len(values) if isinstance(values, list) else type(values).__name__
        raise ProtocolViolation(f"{where}.values must be a list of {NUM_JOINTS} numbers, got {got}")
    parsed = []
    for i, v in enumerate(values):
        x = _number(v, f"{where}.values[{i}]")
        if abs(x) > MAX_JOINT_DELTA[i]:
            raise ProtocolViolation(f"{where}.values[{i}]={x} exceeds max {MAX_JOINT_DELTA[i]}")
        parsed.append(x)
    if "gripper" not in raw:
        raise ProtocolViolation(f"{where}.gripper is required")
    gripper = _number(raw["gripper"], f"{where}.gripper")
    if not 0.0 <= gripper <= 1.0:
        raise ProtocolViolation(f"{where}.gripper={gripper} outside [0, 1]")
    return JointDelta(tuple(parsed), gripper)


def parse_action(raw: Any) -> list[JointDelta]:
    """Validate a joint_delta or action_chunk; returns the substeps in order. Never clamps."""
    if isinstance(raw, dict) and raw.get("type") == "action_chunk":
        _check_keys(raw, {"type", "actions", "chunk_len"}, "action")
        actions = raw.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_CHUNK:
            got = len(actions) if isinstance(actions, list) else type(actions).__name__
            raise ProtocolViolation(f"action.actions must be a list of 1..{MAX_CHUNK} joint_delta, got {got}")
        chunk_len = raw.get("chunk_len", len(actions))
        if isinstance(chunk_len, bool) or chunk_len != len(actions):
            raise ProtocolViolation(f"action.chunk_len={chunk_len!r} does not match {len(actions)} actions")
        return [_parse_joint_delta(a, f"action.actions[{i}]") for i, a in enumerate(actions)]
    return [_parse_joint_delta(raw, "action")]


def _jpeg_b64(frame: np.ndarray, quality: int, *, subsampling: int = 0) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(
        buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=False, progressive=False
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_views(frames: dict[str, np.ndarray], jpeg_quality: int) -> dict[str, str]:
    """JPEG-encode audience cameras (4:2:0, capped at q76). Never enters the policy observation."""
    q = min(jpeg_quality, 76)
    return {name: _jpeg_b64(frame, q, subsampling=2) for name, frame in frames.items()}


def _goal_color(key: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(key.encode()).digest()
    return tuple(70 + byte % 140 for byte in digest[:3])  # type: ignore[return-value]


# Tasks that have a finished scene register a real 640×480 frame here.
# Keys that are absent stay the flat stand-in, so placeholders keep their border.
_GOAL_FRAMES: dict[str, np.ndarray] = {}
# Goals registered but not rendered yet (an env with render off): rendered on first read.
_GOAL_SOURCES: dict[str, Callable[[], np.ndarray]] = {}


def register_goal_image(key: str, frame: np.ndarray) -> None:
    """Remember the finished-scene picture for this task. Sent on every /act."""
    if frame.shape != (GOAL_HEIGHT, GOAL_WIDTH, 3) or frame.dtype != np.uint8:
        raise ValueError(f"goal for {key} must be uint8 {GOAL_HEIGHT}x{GOAL_WIDTH}x3, got {frame.dtype} {frame.shape}")
    _GOAL_SOURCES.pop(key, None)
    _GOAL_FRAMES[key] = np.array(frame, copy=True)
    _drop_cached(key)


def register_goal_source(key: str, render: Callable[[], np.ndarray]) -> None:
    """Like register_goal_image, but `render()` makes the picture on the first read of `key`."""
    _GOAL_FRAMES.pop(key, None)
    _GOAL_SOURCES[key] = render
    _drop_cached(key)


def _drop_cached(key: str) -> None:
    for cached in [item for item in _GOAL_CACHE if item[0] == key]:
        del _GOAL_CACHE[cached]


def goal_image(key: str) -> np.ndarray:
    """640×480 picture of the finished scene, or a flat stand-in if none was registered."""
    source = _GOAL_SOURCES.get(key)
    if source is not None:
        register_goal_image(key, source())
    frame = _GOAL_FRAMES.get(key)
    if frame is not None:
        return frame
    blank = np.empty((GOAL_HEIGHT, GOAL_WIDTH, 3), dtype=np.uint8)
    blank[:, :] = _goal_color(key)
    blank[18:28, 18:-18] = 240
    blank[-28:-18, 18:-18] = 240
    blank[18:-18, 18:28] = 240
    blank[18:-18, -28:-18] = 240
    return blank


_GOAL_CACHE: dict[tuple[str, int], str] = {}


def goal_jpeg(key: str, jpeg_quality: int) -> str:
    cached = _GOAL_CACHE.get((key, jpeg_quality))
    if cached is None:
        cached = _jpeg_b64(goal_image(key), jpeg_quality)
        _GOAL_CACHE[(key, jpeg_quality)] = cached
    return cached


def encode_obs(task: TaskSpec, step: int, raw: RawObs, jpeg_quality: int) -> dict[str, Any]:
    # instruction and goal are on step 0 and every later step. success is not.
    return {
        "task_id": task.policy_task_id(),
        "instruction": task.sentence(),
        "goal": goal_jpeg(task.goal_id(), jpeg_quality),
        "step": step,
        "schema_version": SCHEMA_VERSION,
        "images": {name: _jpeg_b64(frame, jpeg_quality) for name, frame in raw.images.items()},
        "proprio": {
            "joint_pos": raw.joint_pos.tolist(),
            "joint_vel": raw.joint_vel.tolist(),
            "gripper": float(raw.gripper),
            "ee_pos": raw.ee_pos.tolist(),
            "ee_quat": raw.ee_quat.tolist(),
        },
    }
