"""The real-arm cell the simulator reproduces: `assets/cell/cell.json`, in the robot base frame.

The robot base frame is the UR `base_link`: origin at the base centre on the mounting plane, z up. The
UR12e stands on its own pedestal beside the table, so the table top is below that plane:
`table_z_in_base` (-Delta h), the one place that number lives. The simulator's world is the base frame
raised by Delta h, so the table top is world z = 0 as the tasks, spawns, seats and scoring assume, and
the robot base stands at world (0, 0, BASE_Z). Observations (`ee_pos`, `ee_quat`) are in the base frame.

engine/scripts/convert_real_cell.py writes cell.json from the team's scans of the cell (table, pedestal,
fixture colour and pose, the real_cell room's placement); engine/scripts/real_cell_calibrate.py rewrites
it from calibration (camera solves, touch-offs). Until a touch-off, the scan -> base transform and so
the side cameras are provisional.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CELL_DIR = Path(__file__).resolve().parent / "assets" / "cell"
CELL: dict = json.loads((CELL_DIR / "cell.json").read_text())
if CELL.get("version") != 1:
    raise ValueError(f"{CELL_DIR / 'cell.json'}: unsupported version {CELL.get('version')!r}")

BASE_Z = -float(CELL["table_z_in_base"])  # the robot base plane above the table top (world z)
TABLE = CELL["table"]
FLOOR_Z = -float(TABLE["height"])  # world z of the floor
PEDESTAL = CELL["pedestal"]
FIXTURE_RGBA = tuple(CELL["fixture"]["rgba"])
FIXTURE_NOMINAL = CELL["fixture"]["nominal"]  # {"xy", "yaw_deg"} in the base frame
SIDE_FOVY = float(CELL["cameras"]["left"]["fovy"])
WRIST_FOVY = float(CELL["cameras"]["wrist"]["fovy"])


def world(p) -> np.ndarray:
    """Base-frame point -> world."""
    return np.asarray(p, float) + [0.0, 0.0, BASE_Z]


def side_cameras() -> dict[str, tuple[list[float], list[float], float]]:
    """name -> (world position, xyaxes, fovy) of the tripod cameras."""
    return {name: (world(cam["pos"]).tolist(), list(cam["xyaxes"]), float(cam["fovy"]))
            for name, cam in CELL["cameras"].items() if name != "wrist"}


def scan_to_world() -> tuple[np.ndarray, np.ndarray]:
    """(rotation, translation) placing the scanned room (scan frame) in the world."""
    s = CELL["scan_to_robot"]
    a = math.radians(s["yaw_deg"])
    rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    scan_table_z = json.loads((CELL_DIR / "scan_measurements.json").read_text())["table"]["top_z"]
    return rot, np.array([*s["xy"], -scan_table_z])
