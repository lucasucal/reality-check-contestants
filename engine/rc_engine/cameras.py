"""Policy camera sets.

three_view is the only policy camera configuration: one wrist camera and two
side cameras. Do not substitute one view for another.

The wrist and both side cameras are Orbbec Gemini 336 RGB streams: 640x360
(16:9), datasheet field of view 86° x 55° (H x V). The side cameras stand on
tripods just off the left (+y) and right (-y) table edges, outside the UR12e
reach envelope where the room allows, and look down at the fixture and the
spawn area.
"""
from __future__ import annotations

from .embodiment import WRIST_CAMERA, WRIST_CAMERA_FOVY, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH

# Policy frame size per camera, (width, height).
GEMINI_336_SIZE = (WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT)
SIDE_CAMERA_FOVY = WRIST_CAMERA_FOVY
FRAME_SIZES: dict[str, tuple[int, int]] = {
    WRIST_CAMERA: GEMINI_336_SIZE,
    "left": GEMINI_336_SIZE,
    "right": GEMINI_336_SIZE,
}

# Tripod Gemini 336s, world metres: (eye, look-at point). The table top is z=0
# and spans x -0.24..1.20, y ±0.52; the robot base is at the origin.
SIDE_CAMERA_POSES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    # 1.41 m above the floor, 0.14 m off the +y edge; sees the tray side-on.
    "left": ((0.68, 0.66, 0.66), (0.73, -0.04, 0.03)),
    # 1.40 m above the floor, past the -y edge and beyond the fixture, outside the reach envelope.
    "right": ((1.02, -0.62, 0.65), (0.73, -0.04, 0.03)),
}

CAMERA_SETS: dict[str, tuple[str, ...]] = {
    "three_view": ("wrist", "left", "right"),
}


def cameras_for(configuration: str) -> tuple[str, ...]:
    try:
        return CAMERA_SETS[configuration]
    except KeyError:
        known = ", ".join(sorted(CAMERA_SETS))
        raise ValueError(f"unknown camera configuration {configuration!r}; expected {known}") from None


def frame_size(camera: str) -> tuple[int, int]:
    """(width, height) of that camera's policy frame."""
    return FRAME_SIZES[camera]
