"""The class runtime and program examples share the same policy request contract."""

from __future__ import annotations

import math
import re
from typing import Any

CONTRACT = "v0"
SCHEMA_VERSION = 3
CAPABILITIES = {"chunking": True, "caching": False}
CAMERA_VIEWS = ("wrist", "left", "right")
MAX_JOINT_DELTA = 0.05
_EPISODE_ID = re.compile(r"[!-~]+")
_PROPRIO_VECTORS = {"joint_pos": 6, "joint_vel": 6, "ee_pos": 3, "ee_quat": 4}


def check_manifest_schema(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    version = manifest.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"class manifest must declare schema_version {SCHEMA_VERSION}; got {version!r}")


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def observation_error(observation: Any) -> str | None:
    """Report missing or incompatible inputs before calling a policy.

    The observation producer validates JPEG contents and camera roles. The
    policy server checks the envelope without decoding the six frames again.
    """
    if not isinstance(observation, dict):
        return "request body must be a JSON object"
    version = observation.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        return f"schema_version must be {SCHEMA_VERSION}"
    episode_id = observation.get("episode_id")
    if not isinstance(episode_id, str) or not _EPISODE_ID.fullmatch(episode_id):
        return "episode_id must be a nonempty printable ASCII string without spaces"
    step = observation.get("step")
    if type(step) is not int or step < 0:
        return "step must be a nonnegative integer"
    for field in ("task_id", "instruction"):
        value = observation.get(field)
        if not isinstance(value, str) or not value.strip():
            return f"{field} must be a nonempty string"
    for field in ("images", "goal"):
        views = observation.get(field)
        if not isinstance(views, dict) or views.keys() != set(CAMERA_VIEWS):
            return f"{field} must contain wrist, left and right JPEGs"
        for view in CAMERA_VIEWS:
            payload = views[view]
            if not isinstance(payload, str) or not payload.startswith("/9j/"):
                return f"{field}.{view} must contain a base64 JPEG"
    proprio = observation.get("proprio")
    if not isinstance(proprio, dict):
        return "proprio must be an object"
    for field, size in _PROPRIO_VECTORS.items():
        values = proprio.get(field)
        if not isinstance(values, list) or len(values) != size or not all(map(_finite_number, values)):
            return f"proprio.{field} must contain {size} finite numbers"
    gripper = proprio.get("gripper")
    if not _finite_number(gripper) or not 0 <= gripper <= 1:
        return "proprio.gripper must be between 0 and 1"
    if not math.isclose(math.hypot(*proprio["ee_quat"]), 1, abs_tol=1e-3):
        return "proprio.ee_quat must be a unit quaternion in wxyz order"
    return None
