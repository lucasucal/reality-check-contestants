"""Policy camera sets.

three_view is the only policy camera configuration: one wrist camera and two
side cameras. Do not substitute one view for another.

All three stream 640x360 (16:9) RGB. The side cameras are the real cell's Orbbec
Gemini 336s on their tripods across the table from the robot (54.9° vertical from
the factory intrinsics), at the poses calibrated against the ChArUco board
(cell.json); the wrist camera is the Gemini 335Lg on the bracket (61.1° vertical).
"""
from __future__ import annotations

from .cell import SIDE_FOVY
from .embodiment import WRIST_CAMERA, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH

# Policy frame size per camera, (width, height).
GEMINI_336_SIZE = (WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT)
SIDE_CAMERA_FOVY = SIDE_FOVY
FRAME_SIZES: dict[str, tuple[int, int]] = {
    WRIST_CAMERA: GEMINI_336_SIZE,
    "left": GEMINI_336_SIZE,
    "right": GEMINI_336_SIZE,
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
