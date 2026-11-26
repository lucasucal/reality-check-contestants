"""Published runner-side UR12e command boundary.

Per-joint step limits (STEP_LIMIT_RAD) are the UR12e's maximum joint speeds at 20 Hz: 120 °/s base and shoulder,
180 °/s elbow and wrists (0.10472 / 0.15708 rad per step). The simulator then scales down a whole step whose TCP or
elbow would move faster than 1.5 m/s (the UR12e's default Normal-mode safety limits); it reports those steps in
the /step `info.speed_scaled` and they are counted here as the "tcp_speed" / "elbow_speed" limits.

Values that are out of range are clamped. An action that cannot be clamped (wrong shape, non-finite, ...) is
answered with ONE hold-position step (zero joint delta, previous gripper target); the episode continues.
"""
from __future__ import annotations

import math
import re
from typing import Any


class MotionViolation(ValueError):
    """An action that cannot be clamped. `reason` is one of HOLD_REASONS (the hold-step category)."""

    def __init__(self, message: str, reason: str = 'bad_field') -> None:
        super().__init__(message)
        self.reason = reason


# Why a step was held. Fixed ids; the text is the team-facing notice wording (never a value from the action).
HOLD_REASONS: dict[str, str] = {
    'wrong_joint_count': 'an action did not have exactly six joint values',
    'non_finite': 'a value was NaN or infinite',
    'overflow': 'a number was too large to read',
    'invalid_json': 'the response was not valid JSON or not a JSON object',
    'bad_chunk': 'an action_chunk did not hold 1 to 16 actions or its chunk_len did not match',
    'missing_joint_delta': 'the response had no joint_delta action',
    'echo_mismatch': 'the response did not echo episode_id and step in the body and headers',
    'engine_rejected': 'the simulator rejected the action',
    'both_forms': 'the response mixed a single joint_delta with an action_chunk',
    'bad_field': 'an action had an unknown field or a value of the wrong type',
}
# PolicyFailure.held_reason / MotionViolation.reason -> hold category (identity for the motion reasons).
# Every reason clamp_action raises. Each is a fixed string about the shape of the team's own action
# (never a value from it), so it is safe as a team-facing failure_detail.
MOTION_REASONS = frozenset({
    'unknown action field', 'expected finite number', 'action must be an object',
    'chunk must contain 1 to 16 actions', 'chunk_len does not match actions', 'expected joint_delta',
    'expected six joint deltas', 'both joint_delta and action_chunk fields',
})
MOTION_REASON = "action_shape"  # error-surfacing failure_reason for motion_limit_violation (warmup/legacy callers)

# Which limit clamped a value (CS-B-10 / CS-AC-10). Fixed ids; the text is the team-facing wording. Counts are per
# applied action element (one simulator step) and per joint, public repetitions only (scoring.evaluation_summary).
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
STEP_LIMIT_RAD = (0.10472, 0.10472, 0.15708, 0.15708, 0.15708, 0.15708)
JOINT_LIMIT_RAD = 6.28319
ELBOW_LIMIT_RAD = 2.792526803190927
CLAMP_LIMITS: dict[str, str] = {
    "step_increment": "per-step increment (more than 0.10472 rad base and shoulder, 0.15708 rad elbow and wrists)",
    "joint_limit": "joint limit (target beyond +-6.28319 rad)",
    "elbow_range": "elbow range (target beyond +-2.7925 rad)",
    "gripper_range": "gripper range (outside 0 to 1)",
    "tcp_speed": "TCP speed (over 1.5 m/s: the simulator scaled the whole step down)",
    "elbow_speed": "elbow speed (over 1.5 m/s: the simulator scaled the whole step down)",
}
SPEED_SCALED = {"tcp_speed": ("tcp_speed", "tcp"), "elbow_speed": ("elbow_speed", "elbow")}
# The joints each limit can name ("gripper" for the gripper range).
CLAMP_JOINTS: dict[str, tuple[str, ...]] = {
    "step_increment": JOINT_NAMES,
    "joint_limit": tuple(j for j in JOINT_NAMES if j != "elbow"),
    "elbow_range": ("elbow",),
    "gripper_range": ("gripper",),
    "tcp_speed": ("tcp",),
    "elbow_speed": ("elbow",),
}
_CLAMP_TOLERANCE = 1e-12  # float noise below this is not a clamp (a float32 overshoot of ~1e-9 is)


def add_clamps(store: dict[str, dict[str, int]], hits: Any) -> None:
    """Add one applied element's (limit, joint) hits to a {limit: {joint: steps}} counter."""
    for limit, joint in hits:
        per = store.setdefault(limit, {})
        per[joint] = per.get(joint, 0) + 1


def add_speed_scaled(store: dict[str, dict[str, int]], info: Any) -> int:
    """Count the engine's `info.speed_scaled` {"tcp_speed": n, "elbow_speed": m} into `store`; returns n + m."""
    scaled = info.get("speed_scaled") if isinstance(info, dict) else None
    total = 0
    if isinstance(scaled, dict):
        for key, (limit, joint) in SPEED_SCALED.items():
            n = scaled.get(key)
            if type(n) is int and n > 0:
                per = store.setdefault(limit, {})
                per[joint] = per.get(joint, 0) + n
                total += n
    return total


def clamp_label(limit: str, joint: str) -> str:
    """"wrist_1 joint limit", "shoulder_pan per-step increment", "elbow range", "gripper range", "TCP speed",
    "elbow speed"."""
    if limit == "step_increment":
        return f"{joint} per-step increment"
    if limit == "joint_limit":
        return f"{joint} joint limit"
    return {"elbow_range": "elbow range", "tcp_speed": "TCP speed", "elbow_speed": "elbow speed"}.get(limit, "gripper range")


def clamp_lines(clamps: dict[str, dict[str, int]], steps: int) -> list[str]:
    """Team-facing lines, e.g. "wrist_1 joint limit: 312 of 1,200 steps" (fixed text and counts only)."""
    lines = []
    for limit in CLAMP_LIMITS:
        for joint in CLAMP_JOINTS[limit]:
            n = (clamps.get(limit) or {}).get(joint, 0)
            if n:
                lines.append(f"{clamp_label(limit, joint)}: {n:,} of {steps:,} steps")
    return lines


_FIELD_NAME = re.compile(r'[^A-Za-z0-9_.-]')
IGNORED_FIELDS_MAX = 8
IGNORED_FIELD_LEN = 32


def motion_team_detail(exc: BaseException) -> str:
    """Team-safe one-liner for a MotionViolation: the fixed reason string, never echoed values."""
    text = str(exc)
    reason = text if text in MOTION_REASONS else "malformed action"
    return f"Rejected action: {reason}."


def hold_action(gripper: float) -> dict:
    """The hold-position step: zero joint delta, the previous gripper target."""
    return {'type': 'joint_delta', 'values': [0.0] * 6, 'gripper': gripper}


def note_ignored(store: list[str], names: Any) -> None:
    """Add unknown top-level field names to `store` (unique, sanitized, capped in count and length)."""
    for name in names:
        clean = _FIELD_NAME.sub('?', str(name))[:IGNORED_FIELD_LEN]
        if clean and clean not in store and len(store) < IGNORED_FIELDS_MAX:
            store.append(clean)


def ignored_top_level(action: Any) -> list[str]:
    """Names of the unknown top-level keys clamp_action drops from an `action_chunk` object (call after it accepted
    the action). A single joint_delta is an action element: unknown keys there are malformed, never ignored."""
    if not isinstance(action, dict) or action.get('type') != 'action_chunk':
        return []
    known = {'type', 'actions', 'chunk_len'}
    found: list[str] = []
    note_ignored(found, [k for k in action if k not in known])
    return found


def clamp_action(action: Any, target: list[float], ignored: list[str] | None = None,
                 hits: list[list[tuple[str, str]]] | None = None) -> tuple[dict, list[float], bool]:
    """Validate and clamp the complete action chunk before any engine step.

    Unknown keys at the top level of an `action_chunk` object are dropped (their names are added to `ignored`);
    unknown keys inside an action element, including a single `joint_delta` action, are malformed (MotionViolation,
    `bad_field`, DEC-3). Unknown keys of the `/act` response body itself are handled by the policy client. Raises MotionViolation before anything is applied.

    `hits` (optional) receives one list per action element of the (limit, joint) pairs that clamped it
    (CLAMP_LIMITS ids, JOINT_NAMES or "gripper"); the caller counts only the elements the simulator applied."""
    def keys(obj: dict, allowed: set[str]) -> None:
        if any(not isinstance(k, str) or k not in allowed for k in obj):
            raise MotionViolation('unknown action field', 'bad_field')

    def finite(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise MotionViolation('expected finite number', 'bad_field')
        try:
            n = float(value)
        except OverflowError:
            raise MotionViolation('expected finite number', 'overflow') from None
        if not math.isfinite(n):
            raise MotionViolation('expected finite number', 'non_finite')
        return n

    if not isinstance(action, dict):
        raise MotionViolation('action must be an object', 'invalid_json')
    chunk = action.get('type') == 'action_chunk'
    if chunk:
        if 'values' in action or 'gripper' in action:
            raise MotionViolation('both joint_delta and action_chunk fields', 'both_forms')
        known = {'type', 'actions', 'chunk_len'}
        actions = action.get('actions')
        if not isinstance(actions, list) or not 1 <= len(actions) <= 16:
            raise MotionViolation('chunk must contain 1 to 16 actions', 'bad_chunk')
        if 'chunk_len' in action and (type(action['chunk_len']) is not int or action['chunk_len'] != len(actions)):
            raise MotionViolation('chunk_len does not match actions', 'bad_chunk')
    else:
        if 'actions' in action or 'chunk_len' in action:
            raise MotionViolation('both joint_delta and action_chunk fields', 'both_forms')
        known = {'type', 'values', 'gripper'}
        actions = [action]
    position = list(target)
    result = []
    element_hits: list[list[tuple[str, str]]] = []
    for item in actions:
        if not isinstance(item, dict) or item.get('type') != 'joint_delta':
            raise MotionViolation('expected joint_delta', 'missing_joint_delta')
        keys(item, {'type', 'values', 'gripper'})  # an action element (single or in a chunk) is strict
        values = item.get('values')
        if not isinstance(values, list) or len(values) != 6:
            raise MotionViolation('expected six joint deltas', 'wrong_joint_count')
        deltas = []
        found: list[tuple[str, str]] = []
        for joint, value in enumerate(values):
            n = finite(value)
            limit = ELBOW_LIMIT_RAD if joint == 2 else JOINT_LIMIT_RAD
            delta = max(-STEP_LIMIT_RAD[joint], min(STEP_LIMIT_RAD[joint], n))
            if abs(n - delta) > _CLAMP_TOLERANCE:
                found.append(('step_increment', JOINT_NAMES[joint]))
            stepped = delta
            next_position = position[joint] + delta
            if next_position > limit:
                delta = limit - position[joint]
                next_position = limit
            elif next_position < -limit:
                delta = -limit - position[joint]
                next_position = -limit
            if abs(stepped - delta) > _CLAMP_TOLERANCE:
                found.append(('elbow_range' if joint == 2 else 'joint_limit', JOINT_NAMES[joint]))
            deltas.append(delta)
            position[joint] = next_position
        raw_grip = finite(item.get('gripper'))
        grip = max(0.0, min(1.0, raw_grip))
        if grip != raw_grip:
            found.append(('gripper_range', 'gripper'))
        element_hits.append(found)
        result.append({'type': 'joint_delta', 'values': deltas, 'gripper': grip})
    output = ({'type': 'action_chunk', 'actions': result, **({'chunk_len': action['chunk_len']} if 'chunk_len' in action else {})}
              if chunk else result[0])
    if ignored is not None:
        note_ignored(ignored, [k for k in action if k not in known])
    if hits is not None:
        hits.extend(element_hits)
    # Compare with a tolerance to avoid marking normal floating point addition as clipping.
    changed = any(abs(float(a) - b) > 1e-12 for original, new in zip(actions, result)
                  for a, b in zip(original['values'], new['values'])) or any(float(a['gripper']) != b['gripper'] for a, b in zip(actions, result))
    return output, position, changed


def validate_action(action: Any, target: list[float]) -> list[float]:
    return clamp_action(action, target)[1]
