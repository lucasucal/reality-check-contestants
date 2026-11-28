"""Observation/action schema — the single source of truth (docs/platform.md §6.3).

v3: UR12e (6 arm joints + gripper). Changing the embodiment changes this file and the
task modules only. /tasks exports the specs built here, and parse_action enforces the
same constants. There is no ee_delta.
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

SCHEMA_VERSION = 3
NUM_JOINTS = 6  # UR12e arm: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
# Policy increments, radians per 50 ms. TCP, elbow and joint-position limits still apply.
MAX_JOINT_DELTA: tuple[float, ...] = (0.05,) * NUM_JOINTS
MAX_STEPS = 1200
TCP_SPEED_LIMIT = 1.5  # m/s
ELBOW_SPEED_LIMIT = 1.5  # m/s
MAX_CHUNK = 16
GOAL_WIDTH = 640
GOAL_HEIGHT = 360
CAMERA_ROLES = ("wrist", "left", "right")


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


POLICY_CAMERAS = tuple(CameraSpec(role, GOAL_WIDTH, GOAL_HEIGHT) for role in CAMERA_ROLES)


def _check_frames(frames: dict[str, np.ndarray]) -> None:
    if frames.keys() != set(CAMERA_ROLES):
        raise ValueError("required camera roles are wrist, left and right")
    for role, frame in frames.items():
        if frame.shape != (GOAL_HEIGHT, GOAL_WIDTH, 3) or frame.dtype != np.uint8:
            raise ValueError(f"{role} must be a {GOAL_WIDTH}x{GOAL_HEIGHT} RGB uint8 frame")


class GoalSet:
    """One staged scene owned by the environment. A headless reset may defer its render."""

    def __init__(self) -> None:
        self.source: Callable[[], dict[str, np.ndarray]] | None = None
        self._frames: dict[str, np.ndarray] | None = None
        self._encoded: dict[int, dict[str, str]] = {}
        self.capture: dict[str, Any] = {}

    def stage(self, render: Callable[[], dict[str, np.ndarray]], capture: dict[str, Any], *, eager: bool) -> None:
        self.source = render
        self.capture = capture
        self._frames = None
        self._encoded.clear()
        if eager:
            self.frames()

    def frames(self) -> dict[str, np.ndarray]:
        if self._frames is None:
            if self.source is None:
                raise ValueError("no finished scene was staged for the required goal cameras")
            frames = self.source()
            _check_frames(frames)
            self._frames = frames
            self.source = None
        return self._frames

    def jpeg(self, quality: int) -> dict[str, str]:
        if quality not in self._encoded:
            self._encoded[quality] = {role: _jpeg_b64(frame, quality) for role, frame in self.frames().items()}
        return self._encoded[quality].copy()

    def metadata(self, quality: int) -> dict[str, Any]:
        return {**self.capture, "views": {
            role: {"width": GOAL_WIDTH, "height": GOAL_HEIGHT, "encoding": "jpeg",
                   "sha256": hashlib.sha256(base64.b64decode(jpeg)).hexdigest()}
            for role, jpeg in self.jpeg(quality).items()}}


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
    # The task list and the active repetition have independently staged goal sets.
    nominal_goal: GoalSet = field(default_factory=GoalSet, compare=False, repr=False)
    episode_goal: GoalSet = field(default_factory=GoalSet, compare=False, repr=False)
    # Public tasks: every phrasing an episode may send (the pool). `instruction` is the current episode's.
    instructions: tuple[str, ...] = ()

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
            body["goal"] = self.nominal_goal.jpeg(jpeg_quality)
            body["goal_metadata"] = self.nominal_goal.metadata(jpeg_quality)
        elif self.wire_id:
            body["wire_id"] = self.wire_id
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
        "ignored_keys": "unknown action_chunk envelope fields except values and gripper; joint_delta fields are strict",
    }


def obs_spec(task: TaskSpec) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "instruction": "string",
        "goal": {c.name: {"encoding": "jpeg", "width": c.width, "height": c.height} for c in POLICY_CAMERAS},
        "images": {c.name: {"encoding": "jpeg", "width": c.width, "height": c.height} for c in task.cameras},
        "proprio": {
            "joint_pos": [NUM_JOINTS],
            "joint_vel": [NUM_JOINTS],
            "gripper": "0 closed .. 1 open",
            "ee_pos": [3],
            "ee_quat": [4],
        },
        "proprio_units": {"joint_pos": "rad", "joint_vel": "rad/s", "ee_pos": "m",
                          "ee_quat": "unit quaternion wxyz, world frame",
                          "ee_frame": "gripperframe palm site in base_link, +z approach, +y finger opening"},
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
    unknown = sorted(str(k) for k in raw if k not in allowed)
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
        parsed.append(min(MAX_JOINT_DELTA[i], max(-MAX_JOINT_DELTA[i], x)))
    if "gripper" not in raw:
        raise ProtocolViolation(f"{where}.gripper is required")
    gripper = _number(raw["gripper"], f"{where}.gripper")
    return JointDelta(tuple(parsed), min(1.0, max(0.0, gripper)))


def parse_action(raw: Any) -> list[JointDelta]:
    """Validate a joint_delta or action_chunk; returns the substeps in order. Clamp finite values before application."""
    if isinstance(raw, dict) and raw.get("type") == "action_chunk":
        if "values" in raw or "gripper" in raw:
            raise ProtocolViolation("action mixes joint_delta and action_chunk fields")
        actions = raw.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_CHUNK:
            got = len(actions) if isinstance(actions, list) else type(actions).__name__
            raise ProtocolViolation(f"action.actions must be a list of 1..{MAX_CHUNK} joint_delta, got {got}")
        chunk_len = raw.get("chunk_len", len(actions))
        if type(chunk_len) is not int or chunk_len != len(actions):
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


def encode_policy_images(frames: dict[str, np.ndarray], jpeg_quality: int) -> dict[str, str]:
    _check_frames(frames)
    return {role: _jpeg_b64(frame, jpeg_quality) for role, frame in frames.items()}


def encode_obs(task: TaskSpec, step: int, raw: RawObs, jpeg_quality: int) -> dict[str, Any]:
    images = encode_policy_images(raw.images, jpeg_quality)
    # instruction and goal are on step 0 and every later step. success is not.
    return {
        "task_id": task.policy_task_id(),
        "instruction": task.sentence(),
        "goal": task.episode_goal.jpeg(jpeg_quality),
        "step": step,
        "schema_version": SCHEMA_VERSION,
        "images": images,
        "proprio": encode_proprio(raw),
    }


def encode_proprio(raw: RawObs) -> dict[str, Any]:
    for name, values, length in (("joint_pos", raw.joint_pos, NUM_JOINTS), ("joint_vel", raw.joint_vel, NUM_JOINTS),
                                 ("ee_pos", raw.ee_pos, 3), ("ee_quat", raw.ee_quat, 4)):
        if values.shape != (length,) or not np.isfinite(values).all():
            raise ValueError(f"proprio.{name} must contain {length} finite numbers")
    if not 0 <= raw.gripper <= 1 or not np.isclose(np.linalg.norm(raw.ee_quat), 1.0, atol=1e-5):
        raise ValueError("proprio requires a measured gripper opening in [0, 1] and unit ee_quat")
    return {
        "joint_pos": raw.joint_pos.tolist(),
        "joint_vel": raw.joint_vel.tolist(),
        "gripper": float(raw.gripper),
        "ee_pos": raw.ee_pos.tolist(),
        "ee_quat": raw.ee_quat.tolist(),
    }
