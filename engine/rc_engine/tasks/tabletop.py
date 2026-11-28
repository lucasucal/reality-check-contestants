"""Fastener tasks on the printed fixture v3 (a tray, three vertical holes, three hex pockets).

A task is data: an instruction pool, the parts on the table (kit part, spawn
region, start poses), the goals (`Goal`: part -> seat, optionally `on`
another part or in a `pose`), decoys (parts that must stay out of given seats)
and an optional order; or 2-4 per-seed `variants` of parts, goals, decoys and
phrasings (seed % count picks one). Public (seen) tasks are one JSON file each
in `assets/tasks/`; hidden (unseen) tasks are the same JSON in a mounted task
pack (`packs.py`). See docs/fastener-tasks.md.

Seats come from `fixture.py`:

* `tray` - the part rests inside the tray volume;
* `hole_vertical` - a bolt stands in the hole: axis within 10° of vertical,
  shank within 3 mm of the hole axis, head resting on (or in the chamfer of)
  the top face, or on the washer the goal puts it `on`; a washer lies flat on
  the top face over the hole, centred within WASHER_CENTRE_TOL;
* `hex_pocket` - a nut lies on the pocket floor (either face down) with its
  flats inside the pocket's fit.

A goal without a seat puts a washer `on` a bolt standing on its head (around
the shank, on the head); a tray goal may ask a bolt to stand on its head
(`pose: standing`).

Every goal also needs its part released (no contact with the robot) and at
rest. Success is every goal met, every decoy clear, held for HOLD_STEPS. A
part in a hole or pocket other than its goal seat scores 0 for that goal.

The goal picture is the scene with every goal part seated. Colliders are
primitive or convex pieces, so the holes are real; part meshes are visual and
give the mass and inertia.
"""
from __future__ import annotations

import functools
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from ..arm_limits import (
    JOINT_STEP_CLAMP,
    SAFETY_LIMITS,
    SafetyMeter,
    SpeedScaler,
    over_limit,
    safety_stop_mode,
)
from .. import cell
from ..cameras import CAMERA_SETS, frame_size
from ..embodiment import (
    ARM_KP,
    ARM_KV,
    GRIPPER_ACTUATOR,
    GRIPPER_JOINT,
    GRIPPERFRAME,
    HOME_Q,
    PHYSICS_DT,
    SERVOJ_LOOKAHEAD,
    robot_spec,
)
from ..env import StepResult, hash_state
from ..schema import (
    NUM_JOINTS,
    CameraSpec,
    JointDelta,
    RawObs,
    TaskSpec,
    GoalSet,
    MAX_STEPS,
)
from ..score import doneness_point, episode_score, measurement_quality, speed_bonus
from . import rooms, variation
from .fixture import (
    BODY_MESH,
    FIXTURE_BODY,
    FIXTURE_ORIGIN,
    FIXTURE_SPEC,
    FOOTPRINT_X,
    FOOTPRINT_Y,
    MM,
    NOMINAL_POSE,
    SEATS,
    FixturePose,
    Seat,
    add_fixture,
    hex_fit_yaw,
    lean_face,
    seats_for,
)
from .variation import Episode, Shape, clear_around, gap, grasp_zone, shape_at, shape_of

SUBSTEPS = 10
CONTROL_HZ = 20
CONTROL_DT = SUBSTEPS * PHYSICS_DT
_LOOKAHEAD_ALPHA = PHYSICS_DT / SERVOJ_LOOKAHEAD
_ARM_KP = np.array(ARM_KP)
_ARM_KV = np.array(ARM_KV)
# Wrist and side cameras are 640x360 (Gemini 336 RGB). See cameras.FRAME_SIZES.
CAMERA_CONFIGURATIONS = {
    name: tuple(CameraSpec(camera, *frame_size(camera)) for camera in cameras)
    for name, cameras in CAMERA_SETS.items()
}
CAMERAS = CAMERA_CONFIGURATIONS["three_view"]
HOLD_STEPS = 6
# A part counts as in the tray while its underside is below the wall tops plus this (a stack).
_TRAY_STACK = 0.03
# A bolt tip this far above a hole's top face, within its mouth, is standing on that hole.
_MOUTH_REST = 0.002
REST_SPEED = 0.08
EPISODE_SECONDS = 60.0  # one scenario minute; also the speed-score budget
# Cylinder geom axis is +Z. This quat turns that axis onto the body's +X.
_CYL_ALONG_X = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
# Body +X (bolt head direction) onto world +Z: head up.
_HEAD_UP_QUAT = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)
# Body +X onto world -Z: a bolt standing on its head, shank up.
_HEAD_DOWN_QUAT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
# Upright: a bolt axis (or a nut/washer axis) within this of vertical.
UPRIGHT_TILT = math.radians(10.0)
# A part counts as lying flat with its rest axis within this of the table's.
_FLAT_TILT = math.radians(15.0)

# Start poses by part shape. `lying`: on its side (bolt) or flat (nut, washer), the default.
# `standing`: a bolt on its head, shank up. `leaning`: a bolt on its head against the fixture block's
# -y face, tilted LEAN_OVER_DEG past the angle it would fall back upright from. `on_edge`: a nut on one
# of its side flats, axis horizontal along the spawn yaw.
START_POSES = {"bolt": ("lying", "standing", "leaning"), "nut": ("lying", "on_edge"), "washer": ("lying",)}
# Below its tip-over angle (`tip_over_angle`) a leaning bolt stands back up. Further past it the face
# and the table carry more side load, and MuJoCo's soft friction lets that load creep: at +8° an M24
# leans ~2° further over 60 s idle (0.6 mm), at +20° 7° (2.5 mm); real PLA on steel would not move.
LEAN_OVER_DEG = (3.0, 8.0)
_LEAN_GAP = 0.0005  # a leaning bolt is put down this far off the table and the wall, then settles
_LEAN_TRIES = 12
# A washer over a hole: its centre within this of the hole axis: the M30 shank's 2 mm play in the DIN
# 125-A M33 washer's 34 mm bore plus its 2.5 mm play in the 35 mm hole (further off, a shank cannot pass
# both the washer and the bore), plus 0.5 mm for contact softness and the washer collider's 16-sided bore.
WASHER_CENTRE_TOL = 0.005
# Pass tolerances of the seat tests, shared by the success test (`_goal_ok`) and the finish quality
# (`goal_measures`). A washer or nut face on its seat face (a hole's top face, a pocket floor, a bolt's
# bearing face): height off it. A bolt's bearing face: height off its rest.
FACE_HEIGHT_TOL = 0.002
BOLT_HEIGHT_TOL = 0.003
# A bolt's tip and bearing face off the hole axis: the hole's radial clearance plus BOLT_AXIS_SLACK, at
# least BOLT_AXIS_MIN_TOL. A washer around a standing bolt's shank: its bore's play plus the same slack.
BOLT_AXIS_MIN_TOL = 0.003
BOLT_AXIS_SLACK = 0.0005
# A nut in a hex pocket: centre within the radial clearance plus NUT_CENTRE_SLACK; flats within the fit's
# yaw (`hex_fit_yaw`) plus NUT_YAW_SLACK.
NUT_CENTRE_SLACK = 0.001
NUT_YAW_SLACK = math.radians(1.0)
# A part resting `on` a standing bolt: the bolt may have moved this far from where it started.
ON_BASE_DRIFT = 0.03

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_ROOM = _ASSETS / "tabletop"
_FASTENERS = _ASSETS / "fasteners"
TASK_DIR = _ASSETS / "tasks"

PART_FRICTION = [1.2, 0.04, 0.001]

# The part shapes are ISO hardware from the SimReady Hardware dataset (assets/fasteners/VENDORED.md);
# every part is grey PLA, as the event kit is printed. Mass = the dataset's part volume x PART_DENSITY:
# solid PLA (1240 kg/m³) at ~20% infill with 3 walls, about 55% of solid. Estimated; replace by
# weighing real prints. Inertia uses the same density.
PART_DENSITY = 680.0
PART_MATERIAL = "pla_grey"
PART_LOOK = {"rgba": [0.60, 0.61, 0.62, 1.0], "specular": 0.3, "shininess": 0.35}

# Task files, and the bounds and views below, are written for the layout they were designed on: the
# fixture at REFERENCE_ORIGIN. Every spawn, bound and view moves with the fixture's nominal position
# (the real cell's, fixture.FIXTURE_ORIGIN) by LAYOUT_SHIFT, so the parts keep their place beside it.
REFERENCE_ORIGIN = (0.66, 0.15)
LAYOUT_SHIFT = (FIXTURE_ORIGIN[0] - REFERENCE_ORIGIN[0], FIXTURE_ORIGIN[1] - REFERENCE_ORIGIN[1])


def _shifted(p: tuple[float, ...]) -> tuple[float, ...]:
    return (p[0] + LAYOUT_SHIFT[0], p[1] + LAYOUT_SHIFT[1], *p[2:])


# Spawn pads must keep a part on the table, inside comfortable reach and off the fixture.
SPAWN_X = (0.40 + LAYOUT_SHIFT[0], 1.05 + LAYOUT_SHIFT[0])
SPAWN_Y = (-0.42 + LAYOUT_SHIFT[1], 0.42 + LAYOUT_SHIFT[1])
SPAWN_REACH = 1.05
_SPAWN_FIXTURE_CLEARANCE = 0.015
# Clutter lands on the parts' side of the fixture (-y), where both side cameras see it.
CLUTTER_REGION = ((0.42 + LAYOUT_SHIFT[0], 1.02 + LAYOUT_SHIFT[0]), (-0.42 + LAYOUT_SHIFT[1], -0.02 + LAYOUT_SHIFT[1]))
_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
HIDDEN_WIRE_IDS = ("1", "2", "3", "4")  # the unseen tasks' ids on the wire, unique across mounted packs


# ------------------------------------------------------------------ fastener kit

@dataclass(frozen=True)
class Cylinder:
    radius: float
    half: float
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    along: str = "z"


@dataclass(frozen=True)
class Part:
    """One kit fastener (geometry; every part is PART_DENSITY grey PLA). Metres."""

    part_id: str
    shape: str  # "nut", "washer", "bolt"
    colliders: tuple[Cylinder, ...]
    across_flats: float = 0.0  # nut, or bolt head
    thickness: float = 0.0  # nut / washer
    hole: float = 0.0  # nut hole / washer inner diameter
    outer_diameter: float = 0.0  # washer
    diameter: float = 0.0  # bolt shank
    length: float = 0.0  # bolt, under the head
    head_height: float = 0.0  # bolt
    mesh_path: Path | None = None
    # Mass properties (body frame), density-free: mass = volume x density, inertia = inertia_unit x
    # density about `com` (ixx iyy izz ixy ixz iyz). From the dataset sidecars; a pack part without them
    # is integrated from its closed mesh.
    volume: float = 0.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia_unit: tuple[float, ...] = ()
    standard: str = ""
    hull_path: Path | None = None  # a nut's collision mesh (its convex hull); the visual mesh without one

    def mass_properties(self, density: float) -> tuple[float, np.ndarray, list[float]]:
        """(mass, centre of mass, full inertia) for a part of this density."""
        if self.inertia_unit:
            return self.volume * density, np.array(self.com), [density * v for v in self.inertia_unit]
        return _mesh_inertia(self.mesh_path or _FASTENERS / f"{self.part_id}.obj", density)

    @property
    def head_local_x(self) -> float:
        """Bolt: centre of the head along body +X."""
        return 0.5 * self.length

    @property
    def under_head_x(self) -> float:
        """Bolt: the head's bearing face along body +X."""
        return 0.5 * (self.length - self.head_height)

    @property
    def tip_x(self) -> float:
        """Bolt: shank end along body +X (negative)."""
        return -0.5 * (self.length + self.head_height)

    @property
    def lying_radius(self) -> float:
        """Body-centre height of a part resting on the table in its spawn pose."""
        if self.shape == "bolt":
            return 0.5 * self.across_flats
        return 0.5 * self.thickness

    @property
    def footprint(self) -> float:
        """Radius of the part's top-view outline around its body origin."""
        if self.shape == "bolt":
            return 0.5 * (self.length + self.head_height)
        if self.shape == "nut":
            return self.across_flats / math.sqrt(3.0)
        return 0.5 * self.outer_diameter


def _along(axis: object) -> str:
    if isinstance(axis, (list, tuple)) and len(axis) == 3:
        ax = [abs(float(a)) for a in axis]
        return "x" if ax[0] >= max(ax[1], ax[2]) else "z"
    return "x" if str(axis).lower() in {"x", "+x"} else "z"


def _load_kit(manifest: Path | None = None) -> dict[str, Part]:
    manifest = manifest or _FASTENERS / "manifest.json"
    doc = json.loads(manifest.read_text())
    kit: dict[str, Part] = {}
    for entry in doc["parts"]:
        part_id = Path(entry["file"]).stem
        cols = tuple(
            Cylinder(float(c["radius"]), float(c["half_height"]), tuple(float(v) for v in c["pos"]),  # type: ignore[arg-type]
                     _along(c.get("axis", "z")))
            for c in entry["colliders"])
        kind = entry["kind"]
        if kind == "nut":
            part = Part(part_id, kind, cols, across_flats=entry["across_flats"], thickness=entry["thickness"],
                        hole=entry["hole"])
        elif kind == "washer":
            part = Part(part_id, kind, cols, thickness=entry["thickness"], hole=entry["inner_diameter"],
                        outer_diameter=entry["outer_diameter"])
        else:
            part = Part(part_id, kind, cols, across_flats=entry["across_flats"], diameter=entry["diameter"],
                        length=entry["length"], head_height=entry["head_height"])
        mass = {}
        if "inertia_per_density" in entry:
            mass = dict(volume=float(entry["volume_m3"]), com=tuple(float(c) for c in entry["com"]),
                        inertia_unit=tuple(float(v) for v in entry["inertia_per_density"]))
        hull = manifest.parent / entry["collision_mesh"] if "collision_mesh" in entry else None
        kit[part_id] = replace(part, mesh_path=manifest.parent / entry["file"], standard=entry.get("standard", ""),
                               hull_path=hull, **mass)
    return kit


KIT: dict[str, Part] = _load_kit()


# ------------------------------------------------------------------ task model

@dataclass(frozen=True)
class Spawn:
    xy: tuple[float, float]
    jitter: tuple[float, float] = (0.01, 0.01)
    yaw: float = 0.0  # radians; bolt: head direction, nut: a flat normal (on edge: the axis), from +x
    yaw_jitter: float = 0.25
    # Start poses (START_POSES) and their weights; an episode draws one when there are several.
    poses: tuple[tuple[str, float], ...] = (("lying", 1.0),)


@dataclass(frozen=True)
class Start:
    """Where a part starts an episode: its start pose, the drawn table xy and yaw, and the body pose."""

    pose: str
    xy: np.ndarray
    yaw: float
    pos: np.ndarray
    quat: np.ndarray

    def shape(self, part: Part) -> Shape:
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.quat)
        return shape_at(part, self.pos, rot.reshape(3, 3))


@dataclass(frozen=True)
class TaskPart:
    name: str  # id inside the task, e.g. "bolt", "decoy"
    part: Part
    spawn: Spawn

    @property
    def body(self) -> str:
        return f"part_{self.name}"


@dataclass(frozen=True)
class Goal:
    """`part` in `seat`, optionally `on` another part, optionally in a `pose`:

    * {part, seat}: the part in a fixture seat (a washer in a hole: lying flat over its mouth);
    * {part, seat: tray, pose: standing}: a bolt standing on its head in the tray;
    * {part, seat: <hole>, on: <washer>}: a bolt in the hole through a washer lying over it, its head
      on the washer;
    * {part, on: <bolt>}: a washer around the shank of a bolt standing on its head, resting on the head.
    """

    part: str
    seat: str = ""
    on: str = ""
    pose: str = ""


@dataclass(frozen=True)
class Decoy:
    part: str
    not_in: tuple[str, ...]


@dataclass(frozen=True)
class TableTask:
    task_id: str
    instruction: str
    parts: tuple[TaskPart, ...]
    goals: tuple[Goal, ...]
    decoys: tuple[Decoy, ...] = ()
    ordered: bool = False
    visibility: str = "public"
    wire_id: str = ""
    description: str = ""
    episode_seconds: float = EPISODE_SECONDS
    # Odd seeds swap the first two parts' spawn regions (xy and jitter; each keeps its yaw).
    spawn_swap: bool = False
    # Per-seed variants (parts, goals, decoys), each a complete task; seed % count picks one. Empty
    # for a single-scene task. A task with variants carries its first variant's parts, goals, decoys.
    variants: tuple[TableTask, ...] = ()
    fixture_spec: dict | None = None
    fixture_mesh: Path | None = None
    seats_map: dict[str, Seat] | None = None
    # Instruction pool (variation.py): the first phrasing is `instruction`, the rest may carry conditions.
    instructions: tuple = ()
    # Clutter pool: kit parts an episode may add (`clutter<i>` bodies).
    clutter: tuple[TaskPart, ...] = ()

    @property
    def seats(self) -> dict[str, Seat]:
        return self.seats_map if self.seats_map is not None else SEATS

    @property
    def pool(self) -> tuple:
        return self.instructions or (self.instruction,)

    def part(self, name: str) -> TaskPart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def variant_index(self, seed: int) -> int:
        """The variant a seed plays: the seed modulo the variant count (0 without variants)."""
        return seed % len(self.variants) if self.variants else 0

    @property
    def scenes(self) -> tuple[TableTask, ...]:
        """The concrete task of every variant (the task itself when it has none)."""
        return self.variants or (self,)

    @classmethod
    def from_json(cls, entry: dict, *, default_visibility: str = "hidden",
                  fixture_spec: dict | None = None, fixture_mesh: Path | None = None,
                  kit: dict[str, Part] | None = None) -> TableTask:
        """Validate one task definition. Raises ValueError with the reason."""
        if not isinstance(entry, dict):
            raise ValueError("task must be a JSON object")
        if entry.get("kind", "tabletop") != "tabletop":
            raise ValueError(f"kind must be 'tabletop', got {entry.get('kind')!r}")
        known = {"task_id", "kind", "visibility", "wire_id", "instruction", "description", "parts", "goals",
                 "decoys", "ordered", "episode_seconds", "notes", "spawn_swap", "variants", "instructions", "clutter"}
        extra = set(entry) - known
        if extra:
            raise ValueError(f"unknown fields {sorted(extra)}")
        if "variants" in entry:
            return cls._from_variants(entry, default_visibility, fixture_spec, fixture_mesh, kit)
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not _ID.match(task_id):
            raise ValueError("task_id must be a lowercase identifier")
        if ("instruction" in entry) == ("instructions" in entry):
            raise ValueError("give one of instruction (a sentence) or instructions (a pool)")
        pool: tuple = ()
        if "instructions" in entry:
            names_ = {p.get("id") for p in entry.get("parts") or [] if isinstance(p, dict)}
            pool = variation.validate_pool(entry["instructions"], names_, "instructions")
            instruction = pool[0]
        else:
            instruction = entry.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        visibility = entry.get("visibility", default_visibility)
        if visibility not in ("public", "hidden"):
            raise ValueError(f"visibility must be 'public' or 'hidden', got {visibility!r}")
        wire_id = str(entry.get("wire_id") or "")
        if visibility == "hidden" and wire_id not in HIDDEN_WIRE_IDS:
            raise ValueError(f"a hidden tabletop task needs wire_id one of {', '.join(HIDDEN_WIRE_IDS)}")
        if wire_id and wire_id not in HIDDEN_WIRE_IDS:
            raise ValueError(f"wire_id must be one of {', '.join(HIDDEN_WIRE_IDS)}, got {wire_id!r}")
        description = entry.get("description", "")
        if not isinstance(description, str):
            raise ValueError("description must be a string")
        seconds = entry.get("episode_seconds", EPISODE_SECONDS)
        if not isinstance(seconds, (int, float)) or not 5 <= seconds <= 120:
            raise ValueError("episode_seconds must be a number of seconds in [5, 120]")

        raw_parts = entry.get("parts")
        if not isinstance(raw_parts, list) or not 1 <= len(raw_parts) <= 4:
            raise ValueError("parts must list 1 to 4 parts")
        parts = tuple(_parse_part(p, kit or KIT) for p in raw_parts)
        seats = seats_for(fixture_spec) if fixture_spec is not None else SEATS
        names = [p.name for p in parts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate part ids {names}")
        _check_spawn_layout(parts)
        spawn_swap = entry.get("spawn_swap", False)
        if not isinstance(spawn_swap, bool):
            raise ValueError("spawn_swap must be a boolean")
        if spawn_swap:
            if len(parts) < 2:
                raise ValueError("spawn_swap needs at least two parts")
            _check_spawn_layout(_swapped(parts))

        raw_goals = entry.get("goals")
        if not isinstance(raw_goals, list) or not raw_goals:
            raise ValueError("goals must list at least one goal")
        goals = [_parse_goal(g, parts, seats) for g in raw_goals]
        goal_parts = [g.part for g in goals]
        if len(set(goal_parts)) != len(goal_parts):
            raise ValueError("a part can have only one goal")
        _check_stacks(goals, parts)
        seats_used = [g.seat for g in goals if g.seat and seats[g.seat].kind != "tray" and not g.on]
        if len(set(seats_used)) != len(seats_used):
            raise ValueError("two goals share one hole or pocket")
        decoys: list[Decoy] = []
        for dcy in entry.get("decoys", []) or []:
            if not isinstance(dcy, dict) or set(dcy) != {"part", "not_in"}:
                raise ValueError(f"decoy must be {{part, not_in}}, got {dcy!r}")
            if dcy["part"] not in names or dcy["part"] in goal_parts or any(g.on == dcy["part"] for g in goals):
                raise ValueError(f"decoy {dcy['part']!r} must be a part without a goal")
            not_in = dcy["not_in"]
            if isinstance(not_in, str):
                not_in = [not_in]
            if not not_in or any(s not in seats for s in not_in):
                raise ValueError(f"decoy not_in must name seats, got {dcy['not_in']!r}")
            decoys.append(Decoy(dcy["part"], tuple(not_in)))
        ordered = entry.get("ordered", False)
        if not isinstance(ordered, bool):
            raise ValueError("ordered must be a boolean")
        clutter = parse_clutter(entry.get("clutter", []), kit or KIT)
        return cls(task_id, instruction.strip(), parts, tuple(goals), tuple(decoys), ordered, visibility,
                   wire_id, description, float(seconds), spawn_swap, (), fixture_spec, fixture_mesh, seats,
                   pool, clutter)

    @classmethod
    def _from_variants(cls, entry: dict, default_visibility: str, fixture_spec: dict | None,
                       fixture_mesh: Path | None, kit: dict[str, Part] | None) -> TableTask:
        """`variants`: 2-4 objects {parts, goals[, decoys]} in place of the top-level ones. Each is
        validated as a whole task with the shared fields (instruction, order, ...)."""
        raw = entry["variants"]
        if not isinstance(raw, list) or not 2 <= len(raw) <= 4:
            raise ValueError("variants must list 2 to 4 variants")
        shared = {k: v for k, v in entry.items() if k != "variants"}
        clash = {"parts", "goals", "decoys"} & set(shared)
        if clash:
            raise ValueError(f"a task with variants keeps {sorted(clash)} inside each variant, not at the top")
        if shared.get("spawn_swap"):
            raise ValueError("spawn_swap and variants both pick by seed parity; use one of them")
        scenes = []
        own = {"parts", "goals", "decoys", "instructions"}
        for i, variant in enumerate(raw):
            if not isinstance(variant, dict) or not {"parts", "goals"} <= set(variant) <= own:
                raise ValueError(f"variant {i} must be {{parts, goals[, decoys, instructions]}}")
            try:
                scene = dict(shared, **variant)
                if "instructions" in variant:  # the variant's own pool replaces the shared phrasing(s)
                    scene.pop("instruction", None)
                scenes.append(cls.from_json(scene, default_visibility=default_visibility,
                                            fixture_spec=fixture_spec, fixture_mesh=fixture_mesh, kit=kit))
            except ValueError as exc:
                raise ValueError(f"variant {i}: {exc}") from exc
        keys = [(tuple((p.name, p.part.part_id, p.spawn.xy, p.spawn.poses) for p in t.parts), t.goals) for t in scenes]
        if len(set(keys)) != len(keys):
            raise ValueError("two variants are the same")
        return replace(scenes[0], variants=tuple(scenes))

    @classmethod
    def from_pack(cls, entry: dict, pack_dir: Path | None = None) -> TableTask:
        data = dict(entry)
        fixture = data.pop("fixture", None)
        fixture_spec = fixture_mesh = None
        if fixture is not None:
            fixture_spec = json.loads(_pack_path(pack_dir, fixture["spec"]).read_text())
            fixture_mesh = _pack_path(pack_dir, fixture["mesh"])
        manifest_name = data.pop("part_manifest", None)
        kit = dict(KIT)
        if manifest_name is not None:
            extra = _load_kit(_pack_path(pack_dir, manifest_name))
            if set(extra) & set(kit):
                raise ValueError("pack part overrides a public kit part")
            kit.update(extra)
        return cls.from_json(data, default_visibility="hidden", fixture_spec=fixture_spec,
                             fixture_mesh=fixture_mesh, kit=kit)


def _pack_path(pack_dir: Path | None, name: str) -> Path:
    if pack_dir is None or not isinstance(name, str) or not name:
        raise ValueError("pack asset needs a file name and pack directory")
    root = pack_dir.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"pack asset {name!r} is missing or outside the pack")
    return path


def _num_pair(value: object, what: str) -> tuple[float, float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
        raise ValueError(f"{what} must be two numbers")
    return float(value[0]), float(value[1])  # type: ignore[index]


def _parse_part(raw: object, kit: dict[str, Part]) -> TaskPart:
    if not isinstance(raw, dict) or set(raw) - {"id", "part", "spawn"}:
        raise ValueError(f"part must be {{id, part, spawn}}, got {raw!r}")
    name = raw.get("id")
    if not isinstance(name, str) or not _ID.match(name):
        raise ValueError(f"part id must be a lowercase identifier, got {name!r}")
    if raw.get("part") not in kit:
        raise ValueError(f"unknown kit part {raw.get('part')!r}")
    spawn = raw.get("spawn")
    if (not isinstance(spawn, dict) or "xy" not in spawn
            or set(spawn) - {"xy", "jitter", "yaw_deg", "yaw_jitter_deg", "pose"}):
        raise ValueError(f"part {name}: spawn must be {{xy[, jitter, yaw_deg, yaw_jitter_deg, pose]}}")
    kind = kit[raw["part"]].shape
    pose = spawn.get("pose", "lying")
    weights = {pose: 1.0} if isinstance(pose, str) else pose
    if (not isinstance(weights, dict) or not weights or any(p not in START_POSES[kind] for p in weights)
            or not all(isinstance(w, (int, float)) and w > 0 for w in weights.values())):
        raise ValueError(f"part {name}: spawn.pose must be one of {START_POSES[kind]} or {{pose: weight}} of them")
    xy = _shifted(_num_pair(spawn["xy"], f"part {name} spawn.xy"))
    jitter = _num_pair(spawn.get("jitter", [0.01, 0.01]), f"part {name} spawn.jitter")
    if not all(0.0 <= j <= 0.06 for j in jitter):
        raise ValueError(f"part {name}: spawn.jitter must be within [0, 0.06] m")
    yaw = spawn.get("yaw_deg", 0.0)
    yaw_jitter = spawn.get("yaw_jitter_deg", 15.0)
    if not isinstance(yaw, (int, float)) or not isinstance(yaw_jitter, (int, float)) or not 0 <= yaw_jitter <= 180:
        raise ValueError(f"part {name}: yaw_deg must be a number and yaw_jitter_deg within [0, 180]")
    return TaskPart(name, kit[raw["part"]],
                    Spawn(xy, jitter, math.radians(float(yaw)), math.radians(float(yaw_jitter)),
                          tuple((p, float(w)) for p, w in weights.items())))


def _parse_goal(raw: object, parts: tuple[TaskPart, ...], seats: dict[str, Seat]) -> Goal:
    """One goal: {part, seat[, pose]} | {part, seat, on} | {part, on} (see `Goal`)."""
    if not isinstance(raw, dict) or "part" not in raw or set(raw) - {"part", "seat", "on", "pose"} \
            or not ({"seat", "on"} & set(raw)):
        raise ValueError(f"goal must be {{part, seat[, on, pose]}} or {{part, on}}, got {raw!r}")
    names = [p.name for p in parts]
    goal = Goal(str(raw["part"]), str(raw.get("seat", "")), str(raw.get("on", "")), str(raw.get("pose", "")))
    if goal.part not in names:
        raise ValueError(f"goal part {goal.part!r} is not in parts")
    part = parts[names.index(goal.part)].part
    if goal.seat and goal.seat not in seats:
        raise ValueError(f"unknown seat {goal.seat!r}; seats: {sorted(seats)}")
    seat = seats.get(goal.seat)
    if goal.pose and (goal.pose != "standing" or part.shape != "bolt" or seat is None or seat.kind != "tray"):
        raise ValueError(f"goal {goal.part}: pose 'standing' is a bolt on its head in the tray, got {raw!r}")
    if goal.on:
        if goal.on not in names or goal.on == goal.part:
            raise ValueError(f"goal {goal.part}: on {goal.on!r} must name another part")
        base = parts[names.index(goal.on)].part
        bolt, washer = (part, base) if seat is not None else (base, part)
        if (seat is not None and seat.kind != "hole_vertical") or bolt.shape != "bolt" or washer.shape != "washer":
            raise ValueError(f"goal {goal.part}: 'on' stacks a bolt in a hole on a washer, or a washer on a bolt")
        if not bolt.diameter < washer.hole < bolt.across_flats:
            raise ValueError(f"{washer.part_id} does not fit over {bolt.part_id}'s shank under its head")
    if seat is not None:
        _check_fit(part, seat)
    return goal


def _check_stacks(goals: list[Goal], parts: tuple[TaskPart, ...]) -> None:
    """A bolt on a washer needs the washer's goal over the same hole; a washer on a bolt needs the bolt
    to start standing (always) and to have no goal of its own (it stays where it stands)."""
    by_part = {g.part: g for g in goals}
    spawn = {p.name: p.spawn for p in parts}
    for g in goals:
        if not g.on:
            continue
        base = by_part.get(g.on)
        if g.seat and (base is None or base.seat != g.seat or base.on):
            raise ValueError(f"goal {g.part}: its washer {g.on!r} needs a goal over {g.seat}")
        if not g.seat and (base is not None or spawn[g.on].poses != (("standing", 1.0),)):
            raise ValueError(f"goal {g.part}: bolt {g.on!r} must start standing and have no goal of its own")


CLUTTER_MAX = 4
# Unused clutter waits on a hidden shelf under the tabletop, out of every camera's view.
_SHELF_TOP = -0.30
_SHELF = ((0.30 + LAYOUT_SHIFT[0], 1.05 + LAYOUT_SHIFT[0]), (-0.36 + LAYOUT_SHIFT[1], 0.36 + LAYOUT_SHIFT[1]))


def parse_clutter(raw: object, kit: dict[str, Part], start: int = 0) -> tuple[TaskPart, ...]:
    """`clutter`: up to 4 objects {part}: nuts or washers from the kit (bolts would confuse the
    instructions' 'the bolt' and 'the smaller bolt'). Bodies `clutter<i>`."""
    if not isinstance(raw, list) or len(raw) > CLUTTER_MAX:
        raise ValueError(f"clutter must list at most {CLUTTER_MAX} parts")
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"part"} or item["part"] not in kit:
            raise ValueError(f"clutter[{i}] must be {{part}} naming a kit part, got {item!r}")
        part = kit[item["part"]]
        if part.shape == "bolt":
            raise ValueError(f"clutter[{i}]: bolts are not clutter")
        k = start + i
        parked = (_SHELF[0][0] + 0.05 + 0.12 * (k % 6), _SHELF[1][0] + 0.12 + 0.25 * (k // 6))
        out.append(TaskPart(f"clutter{k}", part, Spawn(parked, (0.0, 0.0), 0.0, 0.0)))
    return tuple(out)


def _swapped(parts: tuple[TaskPart, ...]) -> tuple[TaskPart, ...]:
    """The first two parts trade spawn regions (xy and jitter); yaws stay with their parts."""
    a, b, *rest = parts
    a2 = replace(a, spawn=replace(a.spawn, xy=b.spawn.xy, jitter=b.spawn.jitter))
    b2 = replace(b, spawn=replace(b.spawn, xy=a.spawn.xy, jitter=a.spawn.jitter))
    return (a2, b2, *rest)


def _check_spawn_layout(parts: tuple[TaskPart, ...]) -> None:
    for p in parts:
        (x, y), (jx, jy) = p.spawn.xy, p.spawn.jitter
        reach = p.part.footprint
        if not (SPAWN_X[0] <= x - jx and x + jx <= SPAWN_X[1] and SPAWN_Y[0] <= y - jy and y + jy <= SPAWN_Y[1]):
            raise ValueError(f"part {p.name}: spawn region leaves x {SPAWN_X} / y {SPAWN_Y}")
        if math.hypot(abs(x) + jx, abs(y) + jy) > SPAWN_REACH:
            raise ValueError(f"part {p.name}: spawn is farther than {SPAWN_REACH} m from the base")
        margin = reach + _SPAWN_FIXTURE_CLEARANCE
        if (FOOTPRINT_X[0] - margin < x + jx and x - jx < FOOTPRINT_X[1] + margin
                and FOOTPRINT_Y[0] - margin < y + jy and y - jy < FOOTPRINT_Y[1] + margin):
            raise ValueError(f"part {p.name}: spawn region overlaps the fixture")
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            gap = math.hypot(a.spawn.xy[0] - b.spawn.xy[0], a.spawn.xy[1] - b.spawn.xy[1])
            need = a.part.footprint + b.part.footprint + math.hypot(*a.spawn.jitter) + math.hypot(*b.spawn.jitter)
            if gap < need + 0.01:
                raise ValueError(f"parts {a.name} and {b.name} can spawn touching ({gap:.3f} m apart, need {need + 0.01:.3f})")


def _check_fit(part: Part, seat: Seat) -> None:
    if seat.kind == "tray":
        return
    if seat.kind == "hole_vertical":
        if part.shape == "washer":  # lying flat on the top face, over the hole's chamfered mouth
            if part.outer_diameter <= seat.diameter + 2 * seat.chamfer:
                raise ValueError(f"{part.part_id} would fall into {seat.seat_id}")
            return
        if part.shape != "bolt":
            raise ValueError(f"{seat.seat_id} takes a bolt or a washer, not {part.part_id}")
        if part.diameter >= seat.diameter:
            raise ValueError(f"{part.part_id} shank does not fit {seat.seat_id}")
        if part.across_flats <= seat.diameter:
            raise ValueError(f"{part.part_id} head would fall into {seat.seat_id}")
        if part.length > seat.top_z - seat.floor_z:
            raise ValueError(f"{part.part_id} is longer than {seat.seat_id} is deep")
        return
    if part.shape != "nut":
        raise ValueError(f"{seat.seat_id} takes a nut, not {part.part_id}")
    if hex_fit_yaw(part.across_flats, seat.across_flats) <= math.radians(2.0):
        raise ValueError(f"{part.part_id} does not fit {seat.seat_id}")


def load_task_file(path: Path, *, default_visibility: str) -> TableTask:
    try:
        doc = json.loads(path.read_text())
        return TableTask.from_json(doc, default_visibility=default_visibility)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _public_tasks() -> tuple[TableTask, ...]:
    tasks = [load_task_file(path, default_visibility="public") for path in sorted(TASK_DIR.glob("*.json"))]
    for task in tasks:
        if task.visibility != "public":
            raise ValueError(f"{task.task_id}: assets/tasks holds public tasks only; hidden tasks go in a pack")
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate public task ids {ids}")
    return tuple(sorted(tasks, key=lambda t: (t.wire_id or "9", t.task_id)))


PUBLIC_TASKS: tuple[TableTask, ...] = _public_tasks()
TASKS: dict[str, TableTask] = {task.task_id: task for task in PUBLIC_TASKS}
PUBLIC_IDS: tuple[str, ...] = tuple(TASKS)


# ------------------------------------------------------------------ scene

def _look(eye: tuple[float, float, float], target: tuple[float, float, float]) -> list[float]:
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def _material(spec: mujoco.MjSpec, name: str, texture: str, *, specular: float, repeat: tuple[float, float],
              uniform: bool) -> None:
    mat = spec.add_material(name=name)
    mat.specular = specular
    mat.shininess = 0.4
    mat.reflectance = 0.0
    mat.texrepeat = list(repeat)
    mat.texuniform = uniform
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture


WORK_FOCUS = _shifted((0.74, 0.0, 0.04))  # between the fixture and the spawn pads
ORBIT_EYE = _shifted((1.55, -0.55, 0.72))
ORBIT_TARGET = _shifted((0.76, 0.02, 0.08))
# Simulator-only capture pose. The mounted wrist looks down on the fixture from about 0.61 m.
GOAL_ARM_Q = (-0.5729, -1.8367, 1.9775, -1.9350, -1.1251, 1.8863)


def _quat_z(deg: float) -> list[float]:
    a = math.radians(deg)
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def _room(spec: mujoco.MjSpec) -> None:
    """The real cell (cell.py): the robot on its pedestal beside the measured table, the cameras."""
    image = mujoco.mjtTexture.mjTEXTURE_2D
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     file=str(_ROOM / "workshop_sky.png"), gridsize=[2, 3], gridlayout="RLUDFB......")
    tab = cell.TABLE
    spec.add_texture(name="tex_table", type=image, file=str(cell.CELL_DIR / tab["texture"]))
    _material(spec, "table_mat", "tex_table", specular=float(tab["specular"]), repeat=(1 / tab["tile_m"],) * 2,
              uniform=True)
    mat = spec.material("table_mat")
    mat.shininess, mat.reflectance = float(tab["shininess"]), float(tab["reflectance"])
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.40, 0.40, 0.42]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    wb = spec.worldbody
    spec.body("base_link").pos = [0.0, 0.0, cell.BASE_Z]
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, cell.FLOOR_Z], size=[3, 3, 0.05], group=3,
                rgba=[0, 0, 0, 0], friction=[1.1, 0.005, 0.0001])
    (cx, cy), (length, width), hz = tab["centre_xy"], tab["size"], 0.5 * tab["thickness"]
    yaw = math.radians(tab["yaw_deg"])
    wb.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, -hz], quat=_quat_z(tab["yaw_deg"]),
        size=[0.5 * length, 0.5 * width, hz], material="table_mat", friction=[1.1, 0.005, 0.0001])
    leg_half = 0.5 * (-2 * hz - cell.FLOOR_Z)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        lx, ly = sx * (0.5 * length - 0.08), sy * (0.5 * width - 0.08)
        wb.add_geom(
            name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, quat=_quat_z(tab["yaw_deg"]),
            pos=[cx + lx * math.cos(yaw) - ly * math.sin(yaw), cy + lx * math.sin(yaw) + ly * math.cos(yaw),
                 cell.FLOOR_Z + leg_half],
            size=[0.035, 0.035, leg_half], rgba=[0.13, 0.10, 0.08, 1.0], contype=0, conaffinity=0)
    # The pedestal: a square column from the floor to the base plane, its faces along the table's, on a
    # grey floor disc.
    ped = cell.PEDESTAL
    disc_half = 0.5 * ped["disc_height"]
    wb.add_geom(name="pedestal_disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[*ped["disc_xy"], cell.FLOOR_Z + disc_half], size=[ped["disc_radius"], disc_half, 0.0],
                rgba=ped["rgba"]["disc"])
    column_half = 0.5 * (cell.BASE_Z - cell.FLOOR_Z)
    wb.add_geom(name="pedestal_column", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, cell.BASE_Z - column_half],
                quat=_quat_z(tab["yaw_deg"]), size=[0.5 * ped["column_width"]] * 2 + [column_half],
                rgba=ped["rgba"]["column"])
    wb.add_light(
        pos=[1.2, -0.4, 1.8], dir=[-0.35, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True, specular=[0.25, 0.25, 0.25], diffuse=[0.75, 0.75, 0.72])
    # Check cameras only. Each sits past the work area and looks back at 45°.
    span = 0.70
    side = (WORK_FOCUS[0] + span * 0.72, WORK_FOCUS[1] + span * 0.70, WORK_FOCUS[2] + span)
    booth = (WORK_FOCUS[0] + span * 0.72, WORK_FOCUS[1] - span * 0.70, WORK_FOCUS[2] + span)
    wb.add_camera(name="side", pos=list(side), xyaxes=_look(side, WORK_FOCUS), fovy=54)
    wb.add_camera(name="booth", pos=list(booth), xyaxes=_look(booth, WORK_FOCUS), fovy=54)
    # Policy side cameras: the real cell's Gemini 336s on their tripods across the table (cell.json).
    for name, (eye, axes, fovy) in cell.side_cameras().items():
        wb.add_camera(name=name, pos=eye, xyaxes=axes, fovy=fovy)
    # Organizer orbit. The watch page moves this. It is not a policy camera.
    wb.add_camera(name="orbit", pos=list(ORBIT_EYE), xyaxes=_look(ORBIT_EYE, ORBIT_TARGET), fovy=48)
    # The policy wrist camera is hardware: robot_spec() puts the Gemini 336 on tool0.
    # JSON map of name -> {pos, xyaxes, fovy}. Wrist coordinates are in the UR
    # tool0 frame (+z is the tool axis); fixed cameras use world coordinates.
    overrides = json.loads(os.environ.get("RC_CAMERA_POSES", "{}"))
    if not isinstance(overrides, dict):
        raise ValueError("RC_CAMERA_POSES must be a JSON object")
    for name, pose in overrides.items():
        if name not in {"wrist", "left", "right"} or not isinstance(pose, dict):
            raise ValueError(f"invalid camera pose {name!r}")
        cam = spec.camera(name)
        if set(pose) - {"pos", "xyaxes", "fovy"}:
            raise ValueError(f"unknown fields in camera pose {name!r}")
        if "pos" in pose:
            cam.pos = pose["pos"]
        if "xyaxes" in pose:
            cam.alt.xyaxes = pose["xyaxes"]
        if "fovy" in pose:
            cam.fovy = float(pose["fovy"])


def _mesh_inertia(path: Path, density: float) -> tuple[float, np.ndarray, list[float]]:
    """Mass, centre of mass and full inertia of a closed, outward OBJ (signed tetrahedra)."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in fields[1:]])
    triangles = np.asarray(vertices)[np.asarray(faces)]
    volumes = np.linalg.det(triangles) / 6.0
    volume = float(volumes.sum())
    if volume <= 0:
        raise ValueError(f"invalid oriented mesh volume for {path}")
    sums = triangles.sum(axis=1)
    centroid = np.einsum("i,ij->j", volumes, sums) / (4 * volume)
    second = np.einsum("i,ijk,ijl->kl", volumes, triangles, triangles)
    second += np.einsum("i,ij,ik->jk", volumes, sums, sums)
    second = second / 20.0 - volume * np.outer(centroid, centroid)
    inertia = density * (np.trace(second) * np.eye(3) - second)
    return density * volume, centroid, [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                        inertia[0, 1], inertia[0, 2], inertia[1, 2]]


_WASHER_SEGMENTS = 16


def _hex_prism(across_flats: float, centre_x: float, half: float) -> list[float]:
    """Vertices of a bolt head: a hexagonal prism along body +X, flats facing ±z as in the kit mesh."""
    r = across_flats / math.sqrt(3.0)
    verts: list[float] = []
    for x in (centre_x - half, centre_x + half):
        for k in range(6):
            a = k * math.pi / 3
            verts += [x, r * math.cos(a), r * math.sin(a)]
    return verts


def _add_part(spec: mujoco.MjSpec, tp: TaskPart, z: float = 0.05) -> None:
    part = tp.part
    mesh_path = part.mesh_path or _FASTENERS / f"{part.part_id}.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"fastener mesh not found: {mesh_path}")
    mesh_name = f"mesh_{tp.name}"
    spec.add_mesh(name=mesh_name, file=str(mesh_path), scale=[1.0, 1.0, 1.0])
    mat = spec.add_material(name=f"mat_{tp.name}")
    mat.rgba = list(PART_LOOK["rgba"])
    mat.specular = PART_LOOK["specular"]
    mat.shininess = PART_LOOK["shininess"]
    body = spec.worldbody.add_body(name=tp.body, pos=[tp.spawn.xy[0], tp.spawn.xy[1], z])
    # The part's own volume and inertia (dataset sidecars), so collision proxies do not fill holes in the mass.
    mass, com, inertia = part.mass_properties(PART_DENSITY)
    body.explicitinertial = 1
    body.mass = mass
    body.ipos = com
    body.fullinertia = inertia
    joint = body.add_freejoint(name=f"{tp.body}_free")
    joint.damping = [0.002, 0.0, 0.0]  # MuJoCo 4: [linear, quadratic, cubic]
    body.add_geom(
        name=f"{tp.body}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
        material=f"mat_{tp.name}", contype=0, conaffinity=0, density=0)
    for index, col in enumerate(part.colliders):
        kwargs: dict[str, object] = dict(
            name=f"{tp.body}_hit_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=list(col.pos), size=[col.radius, col.half],
            group=3, rgba=[0.0, 0.0, 0.0, 0.0],
            friction=list(PART_FRICTION), density=0, condim=4, solref=[0.01, 1.0],
        )
        if part.shape == "nut":
            # The convex hull: the hex and chamfer match the design, the hole stays filled for contact.
            # MuJoCo 4's multi-CCD (on by default) gives it several contacts on a face; on 3.5.0 it got
            # one and a lying PLA nut sank up to 4.6 mm into the table (tests/test_nut_rest.py). The kit
            # ships the hull of the full-resolution mesh (`collision_mesh`): the decimated visual's hull
            # rested up to 1 mm low.
            hull_name = mesh_name
            if part.hull_path is not None:
                hull_name = f"hull_{tp.name}"
                spec.add_mesh(name=hull_name, file=str(part.hull_path), scale=[1.0, 1.0, 1.0])
            kwargs.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=hull_name)
        elif part.shape == "bolt" and col.along == "x" and abs(col.pos[0] - part.head_local_x) < 1e-6:
            # A hex prism, not the kit's cylinder proxy: a lying bolt rests on a head flat. On a
            # round head the bolt is a cone on the table and rolls in circles about its tip
            # (an 0.57 kg M30x60 bolt drifted 20-40 mm in 10 s after reset).
            hex_name = f"hexhead_{tp.name}"
            spec.add_mesh(name=hex_name, uservert=_hex_prism(part.across_flats, col.pos[0], col.half))
            kwargs.update(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=hex_name, pos=[0.0, 0.0, 0.0])
            body.add_geom(**kwargs)  # type: ignore[arg-type]
            continue
        elif part.shape == "washer":
            # A ring of boxes, not the kit's solid disc: cylinder-on-box contact gives a single
            # point and the disc sank ~5 mm into the table; the ring rests flat and keeps its hole.
            r_in, r_out = 0.5 * part.hole, 0.5 * part.outer_diameter
            rc = 0.5 * (r_in + r_out)
            for k in range(_WASHER_SEGMENTS):
                a = 2 * math.pi * k / _WASHER_SEGMENTS
                kwargs.update(
                    name=f"{tp.body}_hit_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[col.pos[0] + rc * math.cos(a), col.pos[1] + rc * math.sin(a), col.pos[2]],
                    quat=list(_yaw_quat(a)),
                    size=[0.5 * (r_out - r_in), math.tan(math.pi / _WASHER_SEGMENTS) * r_out, col.half])
                body.add_geom(**kwargs)  # type: ignore[arg-type]
            continue
        if col.along == "x":
            kwargs["quat"] = list(_CYL_ALONG_X)
        body.add_geom(**kwargs)  # type: ignore[arg-type]


def _build(task: TableTask, clutter: tuple[TaskPart, ...] = (), with_rooms: bool = True) -> mujoco.MjModel:
    spec = robot_spec()
    spec.option.timestep = PHYSICS_DT
    _room(spec)
    if with_rooms:  # visual only, all hidden; the episode shows one (rooms.RoomSwitch)
        rooms.add_rooms(spec)
    add_fixture(spec, task.fixture_spec or FIXTURE_SPEC, task.fixture_mesh or BODY_MESH)
    for tp in task.parts:
        _add_part(spec, tp)
    if clutter:
        (x0, x1), (y0, y1) = _SHELF
        spec.worldbody.add_geom(
            name="clutter_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, group=3, rgba=[0, 0, 0, 0],
            pos=[0.5 * (x0 + x1), 0.5 * (y0 + y1), _SHELF_TOP - 0.01], size=[0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.01])
        for tp in clutter:
            _add_part(spec, tp, _SHELF_TOP + tp.part.lying_radius)
    try:
        model = spec.compile()
    except ValueError as exc:
        if not with_rooms:
            raise
        warnings.warn(f"the rooms do not compile ({exc}); every episode uses the cube-map backdrop", stacklevel=2)
        return _build(task, clutter, with_rooms=False)
    # No MSAA by default: with multisampling the GB10's EGL shadow pass is not bit-stable, so repetitions of one
    # seed could see different pixels. RC_RENDER_OFFSAMPLES overrides it (organizer option).
    model.vis.quality.offsamples = render_offsamples()
    return model


def render_offsamples() -> int:
    """MSAA samples for offscreen renders: RC_RENDER_OFFSAMPLES, else 0 (bit-stable frames)."""
    raw = os.environ.get("RC_RENDER_OFFSAMPLES", "").strip()
    return int(raw) if raw else 0


def _text(entry: object) -> str:
    return entry if isinstance(entry, str) else entry["text"]  # type: ignore[index]


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


# -90° about x: a nut's axis (body z) onto world +y, its body +y down.
_ROLL_OFF_TABLE = np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0])


_POSE_STRIDE = tuple(math.sqrt(p) % 1.0 for p in (2, 3, 5, 7))


def _draw_pose(poses: tuple[tuple[str, float], ...], seed: int, index: int) -> str:
    """The start pose of the task's `index`-th part for `seed`, by weight. A low-discrepancy sequence
    (seed x an irrational stride, one stride per part), not a random draw: every run of consecutive
    seeds (or of even seeds) gets close to the weighted mix, and two parts' poses are independent."""
    if len(poses) == 1:
        return poses[0][0]
    u = (seed * _POSE_STRIDE[index % len(_POSE_STRIDE)] + 0.5) % 1.0
    edges = np.cumsum([w for _p, w in poses]) / sum(w for _p, w in poses)
    return poses[min(int(np.searchsorted(edges, u, side="right")), len(poses) - 1)][0]


HOME_CLEARANCE = 0.01  # fixture block to the robot's colliders at HOME, in plan


def home_keepout(model: mujoco.MjModel, below: float) -> np.ndarray:
    """World xy points (bounding-box corners and edge midpoints) of every robot collider that reaches
    below `below` with the arm at HOME and the gripper open: where the fixture block may not stand."""
    data = mujoco.MjData(model)
    data.qpos[:NUM_JOINTS] = HOME_Q
    grip = model.joint(GRIPPER_JOINT)
    data.qpos[int(model.jnt_qposadr[grip.id])] = float(model.jnt_range[grip.id][1])
    mujoco.mj_kinematics(model, data)
    root = model.body_rootid[model.body("base_link").id]
    corners = np.array([[sx, sy, sz] for sx in (-1, 0, 1) for sy in (-1, 0, 1) for sz in (-1, 1)], float)
    out = []
    for g in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[g]] != root or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        cen, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        world = (corners * half + cen) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        if world[:, 2].min() < below:
            out.append(world[:, :2])
    return np.vstack(out) if out else np.zeros((0, 2))


@functools.lru_cache(maxsize=None)
def tip_over_angle(part: Part) -> float:
    """A bolt standing on its head and tilted about a head flat edge falls back upright below this angle
    (its centre of mass is still inside the edge)."""
    _mass, com, _inertia = part.mass_properties(1000.0)
    return math.atan2(0.5 * part.across_flats, 0.5 * (part.length + part.head_height) - float(com[0]))


def lean_pose(part: Part, fixture: FixturePose, spec: dict, u: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Body pose of a bolt on its head leaning `theta` from vertical against the block's lean face
    (fixture.lean_face) at `u` along it: a head flat edge (parallel to the face) on the table, the shank
    tip's rim on the face, both `_LEAN_GAP` off so the bolt settles onto them."""
    point, inward, _span = lean_face(spec, fixture, u)
    n = np.append(inward, 0.0)  # horizontal, into the face
    up = np.array([0.0, 0.0, 1.0])
    c, s = math.cos(theta), math.sin(theta)
    x = -c * up - s * n  # body +x, toward the head: down and away from the face
    z = (n + s * x) / c  # the head flat normal toward the face: that flat's bottom edge is the lowest
    full = part.length + part.head_height
    face = np.append(point, 0.0)
    out = s * full - c * 0.5 * (part.across_flats - part.diameter) + _LEAN_GAP
    edge = face - out * n + _LEAN_GAP * up
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, np.cross(z, x), z]).ravel())
    return edge - 0.5 * full * x - 0.5 * part.across_flats * z, quat


# ------------------------------------------------------------------ evaluation

@dataclass(frozen=True)
class Measure:
    """One toleranced measurement of a goal part: `label` names it ("<kit part id> <what>"), `err` against
    the seated pose, `tol` the pass tolerance the success test uses."""

    label: str
    err: float
    tol: float


@dataclass
class GoalState:
    goal: Goal
    pose_ok: bool  # geometric seat test
    released: bool
    at_rest: bool
    wrong_seat: str | None  # another hole/pocket the part is in
    met: bool  # pose_ok and released and at rest and not in a wrong seat
    distance: float  # body origin to the seated pose, metres
    doneness: float
    quality: float = 0.0  # finish quality of the seated pose (`goal_quality`); 0.0 unless pose_ok
    quality_limit: str | None = None  # the measurement that set it


@dataclass
class Evaluation:
    goals: list[GoalState]
    decoys_clear: bool
    order_ok: bool
    success_now: bool  # this step; the hold is applied on top
    doneness: float
    counted: int  # goals met and counted for the score (in order for ordered tasks; 0 after a decoy violation)
    unmet_progress: list[float]  # doneness points of the goals not counted (empty after a decoy violation)
    violations: list[str] = field(default_factory=list)
    quality: float = 0.0  # lowest goal quality this step (score.py)
    quality_limit: str | None = None

    @property
    def goals_met(self) -> int:
        return sum(1 for g in self.goals if g.met)


class TabletopEnv:
    # Caller settings that stay with the env when a reset changes variant (see _use_variant).
    # `profile`: the variation profile a reset without options plays ("public"; tests use "nominal").
    _ENV_ATTRS = ("render", "audience", "max_steps", "_free_cam", "profile")

    def __init__(self, task: TableTask | str) -> None:
        if isinstance(task, str):
            task = TASKS[task]
        # A task with per-seed variants gets one scene (model, data, renderers) per variant, all built
        # here; reset(seed) swaps in the seed's scene. `self.task` is always the current scene's task.
        self.nominal_goal, self.episode_goal = GoalSet(), GoalSet()
        self.task_def = task
        self.variant = 0
        self.nominal_goal.stage(self._lazy_nominal_goals, self._goal_metadata(None), eager=False)
        self.profile = "public"
        self._scenes: dict[int, dict] = {}
        self._episode_config: tuple | None = None
        # Clutter pool: the task's own, plus the mounted eval profile's extra parts for this task.
        self.clutter_pool = task.clutter + self._eval_clutter(task)
        for index in reversed(range(len(task.scenes))):
            self._load_scene(task.scenes[index])
            if index:
                self._scenes[index] = self._scene_state()

    @staticmethod
    def _eval_clutter(task: TableTask) -> tuple[TaskPart, ...]:
        profile = variation.eval_profile()
        if profile is None or task.task_id not in profile.clutter_parts:
            return ()
        kit = dict(KIT)
        if profile.part_manifest is not None:
            kit.update(_load_kit(profile.part_manifest))
        return parse_clutter(list(profile.clutter_parts[task.task_id]), kit, start=len(task.clutter))

    def _scene_state(self) -> dict:
        keep = {"task_def", "variant", "_scenes", "_episode_config", "clutter_pool", *self._ENV_ATTRS}
        return {k: v for k, v in self.__dict__.items() if k not in keep}

    def _use_variant(self, index: int) -> None:
        """Swap in variant `index`'s scene; caller settings and the episode configuration carry over."""
        if index == self.variant:
            return
        if not 0 <= index < len(self.task_def.scenes):
            raise ValueError(f"{self.task_def.task_id} has no variant {index}")
        self._scenes[self.variant] = self._scene_state()
        self.__dict__.update(self._scenes.pop(index))
        self.variant = index
        # Variants share the current episode goal set. Re-stage it when switching models.
        self._goal_episode = None
        if self._episode_config is not None:
            self.configure_episode(*self._episode_config)

    def _load_scene(self, task: TableTask) -> None:
        self.task = task
        self.model = _build(task, self.clutter_pool)
        m = self.model
        self._camera_defaults = (m.cam_pos.copy(), m.cam_quat.copy(), m.cam_fovy.copy())
        self._camera_base = self._camera_defaults
        sky = m.texture("sky").id
        adr, width = int(m.tex_adr[sky]), int(m.tex_width[sky])
        self._sky = (sky, adr, m.tex_data[adr:adr + 6 * width * width * 3].copy().reshape(6, width, width, 3))
        self._sky_turns = 0
        # The furnished rooms (rooms.py): the exterior cube map backs a shown room, the workshop photo the fallback.
        self._room_switch = rooms.RoomSwitch(m)
        self._backdrops = {"cubemap": self._sky[2], "exterior": rooms.exterior_faces(width, rooms.ROOM_DIR)}
        self._backdrop = "cubemap"
        self._visual_defaults = (m.light_dir.copy(), m.light_diffuse.copy(), m.light_specular.copy(),
                                 np.array(m.vis.headlight.ambient), np.array(m.vis.headlight.diffuse),
                                 m.mat_rgba.copy(), m.mat_specular.copy(), m.mat_shininess.copy())
        self.data = mujoco.MjData(m)
        self._goal_data = mujoco.MjData(m)
        self.fixture_mocap = int(m.body_mocapid[m.body(FIXTURE_BODY).id])
        self.ctrl_range = m.actuator_ctrlrange.copy()
        self.tip = m.site(GRIPPERFRAME).id
        every = (*task.parts, *self.clutter_pool)
        self.part_body = {tp.name: m.body(tp.body).id for tp in every}
        self.part_qpos = {tp.name: int(m.jnt_qposadr[m.joint(f"{tp.body}_free").id]) for tp in every}
        self.part_dof = {tp.name: int(m.jnt_dofadr[m.joint(f"{tp.body}_free").id]) for tp in every}
        self._parts = {tp.name: tp for tp in every}
        self._speed = SpeedScaler(m, CONTROL_DT)
        self._safety = SafetyMeter(m, self.part_body)
        self.safety_peaks = dict.fromkeys(SAFETY_LIMITS, 0.0)
        # Opt-in protective stop on the same measurement (arm_limits.safety_stop_mode; raises on a bad value).
        self.enforce_safety_stop = safety_stop_mode() == "enforce"
        self.protective_stop: dict | None = None
        self._home_points: np.ndarray | None = None  # home_keepout, on first use
        self.seats: dict[str, Seat] = task.seats
        self.fixture_pose: FixturePose = NOMINAL_POSE
        self.episode: Episode | None = None
        self.active_clutter: tuple[str, ...] = ()
        self._clutter_decoys: tuple[Decoy, ...] = ()
        self._goal_episode: tuple | None = None
        # First part: the single-part attributes older tools read.
        first = task.parts[0].name
        self.obj_body = self.part_body[first]
        self.free_qpos = self.part_qpos[first]
        self.free_dof = self.part_dof[first]
        robot_root = m.body_rootid[m.body("base_link").id]
        self._robot_geom = np.array([m.body_rootid[m.geom_bodyid[g]] == robot_root for g in range(m.ngeom)])
        self._geom_part = {int(g): name for name, b in self.part_body.items()
                           for g in range(m.ngeom) if m.geom_bodyid[g] == b}
        self.state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        # 640x480 audience views. Other frame sizes (the
        # 640x360 Gemini 336 views) get their own renderer in _sized_renderers.
        self.renderer: mujoco.Renderer | None = None
        self._sized_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.audience = os.environ.get("RC_AUDIENCE_CAMERAS") == "1"
        self.audience_images: dict[str, np.ndarray] = {}
        self._free_cam: mujoco.MjvCamera | None = None
        grip_jnt = m.joint(GRIPPER_JOINT).id
        self.grip_act = m.actuator(GRIPPER_ACTUATOR).id
        self.grip_qadr = int(m.jnt_qposadr[grip_jnt])
        closed_q, open_q = (float(x) for x in m.jnt_range[grip_jnt])
        self.grip_closed_q, self.grip_open_q = closed_q, open_q
        open_ctrl, closed_ctrl = (float(x) for x in m.actuator_ctrlrange[self.grip_act])
        self.grip_open_ctrl, self.grip_closed_ctrl = open_ctrl, closed_ctrl
        self.target = np.array(HOME_Q, dtype=float)
        self.q_cmd = np.array(HOME_Q, dtype=float)
        self.step_count = 0
        self.max_steps: int | None = None
        self.hold = 0
        self.hold_quality: float = 1.0  # finish quality over the current success hold (score.py)
        self.hold_quality_limit: str | None = None
        self.render = True
        self._goal_ready = False
        self._start_dist: list[float] = [0.0] * len(task.goals)
        self._first_met: list[int] = [-1] * len(task.goals)
        self.last_eval: Evaluation | None = None
        budget = min(MAX_STEPS, int(round(task.episode_seconds * CONTROL_HZ)))
        # The task list shows the first phrasing, every variant's pool, and the nominal goal set.
        pool = dict.fromkeys(_text(e) for scene in self.task_def.scenes for e in scene.pool)
        self.spec = TaskSpec(
            task.task_id, task.description or task.instruction, budget, CONTROL_HZ, CAMERAS,
            family="tabletop", visibility=task.visibility, instruction=task.instruction,
            wire_id=task.wire_id, instructions=tuple(pool),
            nominal_goal=self.nominal_goal, episode_goal=self.episode_goal)
        self._start_xy: dict[str, np.ndarray] = {}
        self._show_room(rooms.NOMINAL_ROOM)

    # -------------------------------------------------------------- episode setup

    def configure_episode(self, camera_configuration: str, max_steps: int | None,
                          camera_poses: dict | None = None) -> None:
        if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
            raise ValueError("max_steps must be null or a nonnegative integer")
        if camera_configuration not in CAMERA_CONFIGURATIONS:
            raise ValueError(f"unknown camera configuration {camera_configuration!r}; expected three_view")
        m = self.model
        m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = self._camera_defaults
        for name, pose in (camera_poses or {}).items():
            if name not in {"wrist", "left", "right"} or not isinstance(pose, dict):
                raise ValueError(f"invalid camera pose {name!r}")
            if set(pose) - {"pos", "xyaxes", "fovy"}:
                raise ValueError(f"unknown camera pose fields for {name!r}")
            idx = m.camera(name).id
            if "pos" in pose:
                pos = np.asarray(pose["pos"], dtype=float)
                if pos.shape != (3,) or not np.isfinite(pos).all():
                    raise ValueError("camera pos must contain three finite numbers")
                m.cam_pos[idx] = pos
            if "xyaxes" in pose:
                axes = np.asarray(pose["xyaxes"], dtype=float)
                if axes.shape != (6,) or not np.isfinite(axes).all():
                    raise ValueError("camera xyaxes must contain six finite numbers")
                x, y = axes[:3], axes[3:]
                if np.linalg.norm(x) < 1e-8:
                    raise ValueError("camera x axis is zero")
                x = x / np.linalg.norm(x)
                y = y - np.dot(x, y) * x
                if np.linalg.norm(y) < 1e-8:
                    raise ValueError("camera axes are parallel")
                y = y / np.linalg.norm(y)
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, np.column_stack((x, y, np.cross(x, y))).ravel())
                m.cam_quat[idx] = quat
            if "fovy" in pose:
                fovy = float(pose["fovy"])
                if not np.isfinite(fovy) or not 0 < fovy < 180:
                    raise ValueError("camera fovy must be between 0 and 180 degrees")
                m.cam_fovy[idx] = fovy
        # The episode's tripod jitter (reset) applies on top of these configured poses.
        self._camera_base = (m.cam_pos.copy(), m.cam_quat.copy(), m.cam_fovy.copy())
        if self.episode is not None:
            self._apply_cameras(self.episode)
        self.spec = replace(self.spec, cameras=CAMERA_CONFIGURATIONS[camera_configuration])
        self.max_steps = min(MAX_STEPS, max_steps or MAX_STEPS) if max_steps is not None else None
        self._goal_episode = None
        self._episode_config = (camera_configuration, max_steps, camera_poses)

    def render_pose(self) -> dict | None:
        """Public showcase state. All poses are in the engine world frame."""
        if not self.episode or not self._room_switch.shown or self.spec.visibility != "public":
            return None
        visible_parts = {p.name for p in self.task.parts} | {g.part for g in self.task.goals} | set(self.active_clutter)
        parts = {self.model.body(self.part_body[name]).name: [self._parts[name].part.part_id, PART_MATERIAL]
                 for name in sorted(visible_parts)}
        inactive = {body for name, body in self.part_body.items() if name not in visible_parts}
        return {
            "version": 1, "task": self.task.task_id, "variant": self.variant, "room": self._room_switch.shown,
            "parts": parts,
            "bodies": {self.model.body(i).name: [*self.data.xpos[i].tolist(), *self.data.xquat[i].tolist()]
                       for i in range(1, self.model.nbody) if i not in inactive},
        }

    def replay_snapshot(self) -> dict:
        mujoco.mj_getState(self.model, self.data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        ep = self.episode
        # The state carries the fixture pose (mocap); `episode` re-derives the rest (visuals, goal picture).
        return {"snapshot_version": 2, "state_spec": "mjSTATE_INTEGRATION", "state": self.state.tolist(),
                "step": self.step_count, "target": self.target.tolist(), "q_cmd": self.q_cmd.tolist(),
                "hold": self.hold, "work": 0.0, "reference_work": 0.0,  # energy no longer scored; kept for readers
                "start_dist": list(self._start_dist), "goal_first_met": list(self._first_met),
                "parts": [tp.name for tp in self.task.parts], "variant": self.variant,
                "episode": None if ep is None else {"seed": ep.seed, "profile": ep.profile,
                                                    **({"room": ep.room_option} if ep.room_option else {})},
                "clutter": list(self.active_clutter),
                # The room is re-derived from the episode's seed; recorded to check the replay engine agrees.
                "room": None if ep is None else ep.room, "room_shown": self._room_switch.shown,
                "backdrop": self._backdrop}

    def render_snapshot(self, snapshot: dict) -> dict[str, np.ndarray]:
        """Organizer replay: set a recorded `replay_snapshot()` and render the policy cameras again.

        Uses the same refresh and `_grab` as `_obs`, so the pixels match the frames the policy saw.
        The episode's variation (lights, colours, camera jitter, goal picture) is re-derived from the
        snapshot's `episode` (seed and options). The caller must have applied the episode's camera
        configuration (`configure_episode`) first. Overwrites the live data: use it on a replay
        engine, never on one that is serving an episode.
        """
        variant = snapshot.get("variant", 0)
        if isinstance(variant, bool) or not isinstance(variant, int):
            raise ValueError("snapshot variant must be an integer")
        self._ready_goals(variant)
        m, d = self.model, self.data
        state = np.asarray(snapshot["state"], dtype=float)
        if state.shape != self.state.shape:
            raise ValueError(f"snapshot has {state.size} state values; this model needs {self.state.size}")
        info = snapshot.get("episode")
        if not isinstance(info, dict) or isinstance(info.get("seed"), bool) or not isinstance(info.get("seed"), int):
            raise ValueError("snapshot requires episode seed and variation to render its goal")
        if self.task_def.variant_index(info["seed"]) != variant:
            raise ValueError("snapshot variant does not match its seed")
        options = {"profile": info.get("profile"), **({"room": info["room"]} if "room" in info else {})}
        profile = variation.parse_options(options, internal=True)
        ep, layout = self._draw_episode(info["seed"], profile, options.get("room"))
        recorded = snapshot.get("room")
        if recorded is not None and recorded != ep.room:
            raise ValueError(f"snapshot room {recorded!r}, but this engine draws {ep.room!r} for that episode")
        self._apply_episode(ep)
        for field, actual in (("room_shown", self._room_switch.shown), ("backdrop", self._backdrop)):
            if field in snapshot and snapshot[field] != actual:
                raise ValueError(f"snapshot {field} {snapshot[field]!r}, but this engine renders {actual!r}")
        self._capture_episode_goal(ep, layout)
        mujoco.mj_resetData(m, d)
        mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        self._refresh_scene_kinematics()
        return {cam.name: self._grab(d, cam.name, (cam.width, cam.height)) for cam in self.spec.cameras}

    # -------------------------------------------------------------- poses

    def seated_pose(self, goal: Goal, slot: int = 0, slots: int = 1,
                    data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Body position and quaternion of `goal`'s part sitting in its seat (a washer `on` a bolt: on
        that bolt's head where it stands in `data`, default the live data)."""
        tp = self.task.part(goal.part)
        part = tp.part
        if not goal.seat:  # a washer on a standing bolt's head, around its shank
            under = self.bolt_points(goal.on, data)["under_head"]
            return np.array([under[0], under[1], under[2] + 0.5 * part.thickness]), np.array([1.0, 0.0, 0.0, 0.0])
        seat = self.seats[goal.seat]
        if seat.kind == "hole_vertical" and part.shape == "washer":
            return np.array([seat.xy[0], seat.xy[1], seat.top_z + 0.5 * part.thickness]), _yaw_quat(seat.yaw)
        if seat.kind == "hole_vertical":
            rest = seat.top_z + self.task.part(goal.on).part.thickness if goal.on else seat.top_z - self.head_sink(part, seat)
            return np.array([seat.xy[0], seat.xy[1], rest - part.under_head_x]), np.array(_HEAD_UP_QUAT)
        if seat.kind == "hex_pocket":
            # Nut flats face its body ±x; the pocket's flats face ±y. A 30° yaw lines them up.
            yaw = seat.flat_normal_yaw - math.pi / 2 + math.pi / 6
            return np.array([seat.xy[0], seat.xy[1], seat.floor_z + 0.5 * part.thickness]), _yaw_quat(yaw)
        # Tray: lie flat, spread along the tray's long side (the fixture's y). Bolts lie along it.
        v = -seat.half[1] + (slot + 0.5) * 2 * seat.half[1] / slots
        xy = seat.world((0.0, v))
        if goal.pose == "standing":
            z = seat.floor_z + 0.5 * (part.length + part.head_height)
            return np.array([xy[0], xy[1], z]), _quat_mul(_yaw_quat(seat.yaw), np.array(_HEAD_DOWN_QUAT))
        yaw = (math.pi / 2 if part.shape == "bolt" else 0.0) + seat.yaw
        return np.array([xy[0], xy[1], seat.floor_z + part.lying_radius]), _yaw_quat(yaw)

    @staticmethod
    def head_sink(part: Part, seat: Seat) -> float:
        """How far a seated bolt head sits below the top face, its hex corners on the hole's 45° chamfer."""
        corner = part.across_flats / math.sqrt(3.0)
        return float(np.clip(seat.radius + seat.chamfer - corner, 0.0, seat.chamfer))

    def _tray_slots(self) -> dict[str, tuple[int, int]]:
        tray_goals = [g.part for g in self.task.goals if g.seat == "tray"]
        return {name: (i, len(tray_goals)) for i, name in enumerate(tray_goals)}

    def part_pose(self, name: str, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = self.data if data is None else data
        b = self.part_body[name]
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def bolt_points(self, name: str, data: mujoco.MjData | None = None) -> dict[str, np.ndarray]:
        """World points of a bolt: head centre, bearing face centre, shank tip, and the head direction."""
        pos, rot = self.part_pose(name, data)
        part = self._parts[name].part
        axis = rot[:, 0]
        return {"head": pos + axis * part.head_local_x, "under_head": pos + axis * part.under_head_x,
                "tip": pos + axis * part.tip_x, "axis": axis, "centre": pos}

    # -------------------------------------------------------------- seat tests

    def _goal_ok(self, goal: Goal) -> bool:
        """The geometric test of a goal (see `Goal`): its seat, and its pose or the part it rests on."""
        measures = self.goal_measures(goal)
        return measures is not None and all(m.err <= m.tol for m in measures)

    def goal_measures(self, goal: Goal) -> list[Measure] | None:
        """Every toleranced measurement of `goal`'s parts (error and pass tolerance), or None when a test
        without a tolerance fails (wrong shape, out of the tray, the base bolt touched or moved). `_goal_ok`
        is every error within its tolerance; the finish quality grades the same errors."""
        if not goal.seat:
            return self._on_standing_bolt_measures(goal.part, goal.on)
        seat = self.seats[goal.seat]
        if goal.on:  # a bolt in the hole through its washer: the washer over the hole, the head on it
            washer_pos, _rot = self.part_pose(goal.on)
            rest = float(washer_pos[2]) + 0.5 * self._parts[goal.on].part.thickness
            washer, bolt = self._seat_measures(goal.on, seat), self._bolt_in_hole_measures(goal.part, seat, rest)
            return None if washer is None or bolt is None else washer + bolt
        measures = self._seat_measures(goal.part, seat)
        if measures is not None and goal.pose == "standing":
            measures = [*measures, self._tilt_measure(goal.part, head_down=True)]
        return measures

    def _measure(self, name: str, what: str, err: float, tol: float) -> Measure:
        return Measure(f"{self._parts[name].part.part_id} {what}", float(err), float(tol))

    def _tilt(self, name: str, head_down: bool = False) -> float:
        """Angle (rad) of a bolt's axis off vertical (head up, or `head_down`), or of a nut's or washer's axis
        off vertical, either face up."""
        _pos, rot = self.part_pose(name)
        if self._parts[name].part.shape == "bolt":
            c = float(rot[2, 0]) * (-1.0 if head_down else 1.0)
        else:
            c = abs(float(rot[2, 2]))
        return math.acos(min(1.0, max(-1.0, c)))

    def _tilt_measure(self, name: str, head_down: bool = False) -> Measure:
        return self._measure(name, "tilt", self._tilt(name, head_down), UPRIGHT_TILT)

    def _upright(self, name: str, head_down: bool = False) -> bool:
        """A bolt's axis (head up, or `head_down`), or a nut's or washer's axis, within UPRIGHT_TILT of vertical."""
        return self._tilt(name, head_down) <= UPRIGHT_TILT

    def _lying(self, name: str) -> bool:
        """The part lies flat on the table: a bolt on its side, a nut or washer on a face."""
        pos, rot = self.part_pose(name)
        part = self._parts[name].part
        tilt = abs(float(rot[2, 0])) <= math.sin(_FLAT_TILT) if part.shape == "bolt" else \
            abs(float(rot[2, 2])) >= math.cos(_FLAT_TILT)
        return tilt and float(pos[2]) <= part.lying_radius + 0.008

    def base_ok(self, base: str) -> bool:
        """A bolt something rests `on` still stands on its head where it started (ON_BASE_DRIFT), untouched."""
        pos, _rot = self.part_pose(base)
        return (self._upright(base, head_down=True) and base not in self._contact_parts()
                and float(np.linalg.norm(pos[:2] - self._start_xy.get(base, pos[:2]))) <= ON_BASE_DRIFT)

    def _on_standing_bolt(self, name: str, base: str) -> bool:
        """A washer flat on a standing bolt's head (its underside on the bearing face ±2 mm), its centre
        within the shank's play in its bore + 0.5 mm of the bolt axis; the bolt as in `base_ok`."""
        measures = self._on_standing_bolt_measures(name, base)
        return measures is not None and all(m.err <= m.tol for m in measures)

    def _on_standing_bolt_measures(self, name: str, base: str) -> list[Measure] | None:
        """`_on_standing_bolt` as measurements: the base bolt's tilt (head down), the washer's tilt, its centre
        off the bolt axis and its height off the bearing face. The base's drift from its start and its contact
        stay pass/fail (None); they are not part of the seated pose."""
        pos, _rot = self.part_pose(base)
        if base in self._contact_parts() or \
                float(np.linalg.norm(pos[:2] - self._start_xy.get(base, pos[:2]))) > ON_BASE_DRIFT:
            return None
        washer, bolt = self._parts[name].part, self._parts[base].part
        pos, _rot = self.part_pose(name)
        under = self.bolt_points(base)["under_head"]
        play = 0.5 * (washer.hole - bolt.diameter) + BOLT_AXIS_SLACK
        return [self._tilt_measure(base, head_down=True), self._tilt_measure(name),
                self._measure(name, "position", float(np.linalg.norm(pos[:2] - under[:2])), play),
                self._measure(name, "height", abs(float(pos[2]) - 0.5 * washer.thickness - float(under[2])),
                              FACE_HEIGHT_TOL)]

    def _bolt_in_hole(self, name: str, seat: Seat, rest_z: float) -> bool:
        """Head up within UPRIGHT_TILT; tip and bearing face within the hole clearance + 0.5 mm (at least
        3 mm) of the hole axis; the bearing face within 3 mm of `rest_z`."""
        measures = self._bolt_in_hole_measures(name, seat, rest_z)
        return measures is not None and all(m.err <= m.tol for m in measures)

    def _bolt_in_hole_measures(self, name: str, seat: Seat, rest_z: float) -> list[Measure] | None:
        """`_bolt_in_hole` as measurements: tilt (head up; yaw ignored), position (the farther of tip and
        bearing face off the hole axis) and height (bearing face off `rest_z`). None for a part not a bolt."""
        part = self._parts[name].part
        if part.shape != "bolt":
            return None
        pts = self.bolt_points(name)
        centre = np.array(seat.xy)
        off = max(np.linalg.norm(pts["tip"][:2] - centre), np.linalg.norm(pts["under_head"][:2] - centre))
        axis_tol = max(BOLT_AXIS_MIN_TOL, 0.5 * (seat.diameter - part.diameter) + BOLT_AXIS_SLACK)
        return [self._tilt_measure(name), self._measure(name, "position", float(off), axis_tol),
                self._measure(name, "height", abs(float(pts["under_head"][2]) - rest_z), BOLT_HEIGHT_TOL)]

    def _seat_ok(self, name: str, seat: Seat) -> bool:
        measures = self._seat_measures(name, seat)
        return measures is not None and all(m.err <= m.tol for m in measures)

    def _seat_measures(self, name: str, seat: Seat) -> list[Measure] | None:
        """The seat test of one part as measurements (None when its untoleranced part fails).

        Tray: containment is pass/fail (any pose inside counts), so no measurements. Washer over a hole: tilt
        (either face down), height off the top face, centre off the hole axis. Hole: `_bolt_in_hole_measures`.
        Hex pocket: tilt (either face down), height off the floor, centre off the pocket axis, and flat
        alignment (yaw error to the nearest pocket flat, mod 60°)."""
        tp = self._parts[name]
        part = tp.part
        pos, rot = self.part_pose(name)
        if seat.kind == "tray":
            if part.shape == "bolt":
                pts = self.bolt_points(name)
                check = [pts["head"], pts["tip"], pos]
            else:
                check = [pos]
            return [] if all(seat.in_tray(p) for p in check) and self._low_in_tray(name, seat) else None
        if seat.kind == "hole_vertical":
            if part.shape == "washer":  # flat on the top face over the hole, centred
                return [self._tilt_measure(name),
                        self._measure(name, "height", abs(float(pos[2]) - seat.top_z - 0.5 * part.thickness),
                                      FACE_HEIGHT_TOL),
                        self._measure(name, "position", float(np.linalg.norm(pos[:2] - np.array(seat.xy))),
                                      WASHER_CENTRE_TOL)]
            return self._bolt_in_hole_measures(name, seat, seat.top_z - self.head_sink(part, seat))
        # hex pocket
        if part.shape != "nut":  # either face down: a nut is symmetric in its pocket
            return None
        clearance = 0.5 * (seat.across_flats - part.across_flats)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        # The nut's flat normals are at yaw + k·60°; aligned when one matches a pocket flat normal.
        err = (yaw - seat.flat_normal_yaw) % (math.pi / 3)
        err = min(err, math.pi / 3 - err)
        return [self._tilt_measure(name),
                self._measure(name, "height", abs(float(pos[2]) - (seat.floor_z + 0.5 * part.thickness)),
                              FACE_HEIGHT_TOL),
                self._measure(name, "position", float(np.linalg.norm(pos[:2] - np.array(seat.xy))),
                              clearance + NUT_CENTRE_SLACK),
                self._measure(name, "flat alignment", err,
                              hex_fit_yaw(part.across_flats, seat.across_flats) + NUT_YAW_SLACK)]

    def goal_quality(self, goal: Goal) -> tuple[float, str | None]:
        """Finish quality of one goal: the lowest `measurement_quality` over its measurements and the
        measurement that set it (None at 1.0). 0.0 when the goal's untoleranced tests fail."""
        measures = self.goal_measures(goal)
        if measures is None:
            return 0.0, None
        q, limit = 1.0, None
        for m in measures:
            mq = measurement_quality(m.err, m.tol)
            if mq < q:
                q, limit = mq, m.label
        return q, limit

    def _low_in_tray(self, name: str, seat: Seat) -> bool:
        """The part's underside (centre less its lying radius) is down in the tray, not above it.

        Judged by the bottom, not the centre, so a part resting on another part in the
        tray (a washer on a lying bolt, a bolt on a nut) still counts as in it.
        """
        pos, _rot = self.part_pose(name)
        return float(pos[2]) - self._parts[name].part.lying_radius < seat.top_z + _TRAY_STACK

    def _occupies(self, name: str, seat: Seat) -> bool:
        """Looser test: is the part in (or stuck in) this seat at all?"""
        part = self._parts[name].part
        pos, _rot = self.part_pose(name)
        if seat.kind == "tray":
            return seat.in_tray(pos) and self._low_in_tray(name, seat)
        low = self.bolt_points(name)["tip"] if part.shape == "bolt" else pos
        centre = np.array(seat.xy)
        if seat.kind == "hole_vertical":
            # In the bore or its chamfer, or standing on the mouth of a hole too small for it.
            return (float(np.linalg.norm(low[:2] - centre)) <= seat.radius + seat.chamfer
                    and float(low[2]) < seat.top_z + _MOUTH_REST)
        return float(np.linalg.norm(low[:2] - centre)) <= seat.apothem and float(low[2]) < seat.top_z - 0.002

    def _contact_parts(self) -> set[str]:
        """Parts touching any robot geom (gripper pads, arm, wrist mount) this step."""
        d = self.data
        touching: set[str] = set()
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._robot_geom[g1] and g2 in self._geom_part:
                touching.add(self._geom_part[g2])
            elif self._robot_geom[g2] and g1 in self._geom_part:
                touching.add(self._geom_part[g1])
        return touching

    def evaluate(self) -> Evaluation:
        """Score-relevant state of every goal and decoy, from the live data."""
        d = self.data
        touching = self._contact_parts()
        slots = self._tray_slots()
        goal_states: list[GoalState] = []
        for i, goal in enumerate(self.task.goals):
            b = self.part_body[goal.part]
            # cvel is [rotation, translation] at the body. Free-joint qvel order is not this.
            at_rest = float(np.linalg.norm(d.cvel[b][3:])) <= REST_SPEED
            released = goal.part not in touching
            pose_ok = self._goal_ok(goal)
            wrong = None
            for other in self.seats.values():
                if other.seat_id != goal.seat and other.kind != "tray" and self._occupies(goal.part, other):
                    wrong = other.seat_id
                    break
            met = pose_ok and released and at_rest and wrong is None
            slot, count = slots.get(goal.part, (0, 1))
            dist = self._goal_distance(goal, slot, count)
            # A part to rest on a standing bolt makes no progress once that bolt is knocked over or moved.
            failed = wrong or (not goal.seat and not self.base_ok(goal.on))
            point = 0.0 if failed else doneness_point(dist, self._start_dist[i], pose_ok)
            quality, limit = self.goal_quality(goal) if pose_ok else (0.0, None)
            goal_states.append(GoalState(goal, pose_ok, released, at_rest, wrong, met, dist, point, quality, limit))
        # Clutter on the table follows the decoy rule: in a goal's hole or pocket (or the tray) it zeroes the task.
        violations = [f"decoy {dcy.part} in {s}" for dcy in (*self.task.decoys, *self._clutter_decoys)
                      for s in dcy.not_in if self._occupies(dcy.part, self.seats[s])]
        decoys_clear = not violations
        order_ok = True
        if self.task.ordered:
            for i in range(1, len(goal_states)):
                a, b_ = self._first_met[i - 1], self._first_met[i]
                if b_ >= 0 and (a < 0 or a > b_):
                    order_ok = False
                    violations.append(f"goal {i + 1} before goal {i}")
        doneness = float(np.mean([g.doneness for g in goal_states])) if decoys_clear else 0.0
        counted = sum(1 for g in goal_states if g.met)
        if self.task.ordered and not order_ok:
            counted = 0
            for i, g in enumerate(goal_states):
                if not g.met or (i > 0 and (self._first_met[i - 1] < 0 or self._first_met[i - 1] > self._first_met[i])):
                    break
                counted += 1
        if self.task.ordered and not order_ok:
            counted_idx = set(range(counted))
        else:
            counted_idx = {i for i, g in enumerate(goal_states) if g.met}
        if decoys_clear:
            unmet = [g.doneness for i, g in enumerate(goal_states) if i not in counted_idx]
        else:
            counted, unmet = 0, []
        success_now = all(g.met for g in goal_states) and decoys_clear and order_ok
        worst = min(goal_states, key=lambda g: g.quality)
        return Evaluation(goal_states, decoys_clear, order_ok, success_now, doneness, counted, unmet, violations,
                          worst.quality, worst.quality_limit)

    def _goal_distance(self, goal: Goal, slot: int, count: int, data: mujoco.MjData | None = None) -> float:
        pos, rot = self.part_pose(goal.part, data)
        seat = self.seats.get(goal.seat)
        if seat is not None and seat.kind == "tray":
            # Anywhere in the tray counts: distance to the tray floor box.
            part = self.task.part(goal.part).part
            u, v = seat.local(pos)
            centre = 0.5 * (part.length + part.head_height) if goal.pose == "standing" else part.lying_radius
            lo = np.array([-seat.half[0], -seat.half[1], seat.floor_z])
            hi = np.array([seat.half[0], seat.half[1], seat.floor_z + centre])
            p = np.array([u, v, pos[2]])
            dist = float(np.linalg.norm(p - np.clip(p, lo, hi)))
            if goal.pose == "standing":  # plus how far the tip is from standing straight up over the head
                dist += centre * float(np.linalg.norm(rot[:, 0] - np.array([0.0, 0.0, -1.0])))
            return dist
        ideal, _quat = self.seated_pose(goal, slot, count, data)
        return float(np.linalg.norm(pos - ideal))

    # -------------------------------------------------------------- simulation

    def _lay_out(self, data: mujoco.MjData) -> None:
        """Park the arm at home and put every part at its nominal spawn, in its first start pose."""
        data.qpos[:NUM_JOINTS] = HOME_Q
        data.qpos[self.grip_qadr] = self.grip_open_q
        data.ctrl[:NUM_JOINTS] = HOME_Q
        data.ctrl[self.grip_act] = self.grip_open_ctrl
        data.qvel[:] = 0.0
        for tp in self.task.parts:
            start = self._start(tp, tp.spawn.poses[0][0], np.array(tp.spawn.xy), tp.spawn.yaw, self.fixture_pose)
            self._set_part(data, tp.name, start.pos, start.quat)

    def _start(self, tp: TaskPart, pose: str, xy: np.ndarray, yaw: float, fixture: FixturePose,
               rng: np.random.Generator | None = None) -> Start:
        """The body pose of `tp` starting in `pose` at table point xy and yaw (START_POSES). A leaning
        bolt's place on the block face and its tilt come from `rng` (the middle of the ranges without)."""
        part = tp.part
        if pose == "standing":
            pos = np.array([xy[0], xy[1], 0.5 * (part.length + part.head_height)])
            quat = _quat_mul(_yaw_quat(yaw), np.array(_HEAD_DOWN_QUAT))
        elif pose == "on_edge":  # a side flat down, the axis along yaw
            pos = np.array([xy[0], xy[1], 0.5 * part.across_flats])
            quat = _quat_mul(_yaw_quat(yaw - math.pi / 2), _quat_mul(_ROLL_OFF_TABLE, _yaw_quat(math.pi / 6)))
        elif pose == "leaning":
            x0, x1 = lean_face(self.task.fixture_spec or FIXTURE_SPEC, fixture)[2]
            margin = part.across_flats / math.sqrt(3.0) + 0.01
            lo, hi = LEAN_OVER_DEG
            u = float(rng.uniform(x0 + margin, x1 - margin)) if rng is not None else 0.5 * (x0 + x1)
            over = float(rng.uniform(lo, hi)) if rng is not None else 0.5 * (lo + hi)
            pos, quat = lean_pose(part, fixture, self.task.fixture_spec or FIXTURE_SPEC, u,
                                  tip_over_angle(part) + math.radians(over))
            xy = pos[:2]
        else:
            pos, quat = np.array([xy[0], xy[1], part.lying_radius]), _yaw_quat(yaw)
        return Start(pose, np.asarray(xy, float), float(yaw), pos, quat)

    def _set_part(self, data: mujoco.MjData, name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        adr, dof = self.part_qpos[name], self.part_dof[name]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = quat
        data.qvel[dof:dof + 6] = 0.0

    def _drive(self, data: mujoco.MjData, target: np.ndarray, q_cmd: np.ndarray,
               qd_des: np.ndarray, grip_ctrl: float) -> np.ndarray:
        """One control step of the live servo. `q_cmd` is not the env's array."""
        data.ctrl[self.grip_act] = grip_ctrl
        q_cmd = np.array(q_cmd, dtype=float, copy=True)
        live = data is self.data
        samples = np.zeros(len(SAFETY_LIMITS))
        for _ in range(SUBSTEPS):
            q_cmd += _LOOKAHEAD_ALPHA * (target - q_cmd)
            data.ctrl[:NUM_JOINTS] = np.clip(
                q_cmd + (_ARM_KV / _ARM_KP) * qd_des,
                self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
            mujoco.mj_step(self.model, data)
            if live:
                samples += self._safety.measure(data)
        if live:
            means = dict(zip(SAFETY_LIMITS, (float(v) for v in samples / SUBSTEPS)))
            for key, value in means.items():
                if value > self.safety_peaks[key]:
                    self.safety_peaks[key] = value
            if self.enforce_safety_stop and self.protective_stop is None and (over := over_limit(means)):
                self.protective_stop = {"limit": over[0], "value": round(over[1], 1), "step": self.step_count + 1}
        return q_cmd

    def _home(self) -> None:
        d = self.data
        d.qpos[:NUM_JOINTS] = HOME_Q
        d.qpos[self.grip_qadr] = self.grip_open_q
        d.ctrl[:NUM_JOINTS] = HOME_Q
        d.ctrl[self.grip_act] = self.grip_open_ctrl
        self.q_cmd = np.array(HOME_Q, dtype=float)
        self.target = np.array(HOME_Q, dtype=float)

    def _settle(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self._refresh_scene_kinematics()

    def _refresh_scene_kinematics(self) -> None:
        # mj_step integrates qpos and qvel after computing poses and body velocities.
        # Refresh the fields used by rendering and scoring without rerunning the solver.
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        mujoco.mj_comVel(m, d)

    def goal_scene(self, data: mujoco.MjData) -> None:
        """Arm at home, goal parts seated, other parts at their nominal spawn. Not settled."""
        self._lay_out(data)
        self._seat_goals(data)

    def _seat_goals(self, data: mujoco.MjData) -> None:
        """Put every goal part in its seated pose on `data` (a part `on` another after its base)."""
        mujoco.mj_kinematics(self.model, data)
        slots = self._tray_slots()
        for goal in sorted(self.task.goals, key=lambda g: bool(g.on)):
            pos, quat = self.seated_pose(goal, *slots.get(goal.part, (0, 1)), data=data)
            self._set_part(data, goal.part, pos, quat)
            mujoco.mj_kinematics(self.model, data)

    def _capture_goal(self) -> None:
        """The task list's goal picture: the nominal scene (no variation), every goal part seated."""
        m, d = self.model, self.data
        if self.episode is not None:  # first captured after episodes ran unrendered: back to the nominal scene
            self.seats, self.fixture_pose = self.task.seats, NOMINAL_POSE
            self._default_visuals()
            m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = self._camera_base
            self._set_sky(0)
        m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = self._camera_defaults
        mujoco.mj_resetData(m, d)
        self._home()
        self.goal_scene(d)
        d.qpos[:NUM_JOINTS] = GOAL_ARM_Q
        d.ctrl[:NUM_JOINTS] = GOAL_ARM_Q
        mujoco.mj_forward(m, d)
        self._settle(80)
        self._stage_goals(self.nominal_goal, d, None, eager=True)

    def _ready_goals(self, variant: int) -> None:
        """Capture the task list's goal picture once (the first variant's nominal scene), then swap in
        `variant`'s scene. Every episode also renders its own goal picture (`_capture_episode_goal`).
        With `render` off there is no capture (no GL needed); the first rendered reset makes it."""
        ready = self._goal_ready if self.variant == 0 else self._scenes[0]["_goal_ready"]
        if not ready and self.render:
            self._use_variant(0)
            self._capture_goal()
            self._goal_ready = True
        self._use_variant(variant)

    # -------------------------------------------------------------- episode variation

    def _draw_episode(self, seed: int, profile_name: str, room: str | None = None) -> tuple[Episode, dict]:
        """The seed's layout {part: Start} and its variation (variation.py), in `room` if given (reset
        option) instead of the drawn one. Writes nothing."""
        task = self.task
        profile = variation.profile_named(profile_name)
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = []
        for tp in (_swapped(task.parts) if task.spawn_swap and seed % 2 == 1 else task.parts):
            sp = tp.spawn
            xy = np.array(sp.xy) + rng.uniform(-1.0, 1.0, size=2) * np.array(sp.jitter)
            yaw = sp.yaw + float(rng.uniform(-sp.yaw_jitter, sp.yaw_jitter))
            draws.append((tp, xy, yaw))
        index = {tp.name: i for i, tp in enumerate(task.parts)}
        poses = {tp.name: _draw_pose(tp.spawn.poses, seed, index[tp.name]) for tp, _xy, _yaw in draws}
        pose_rng = variation.stream(seed, "pose")  # where a leaning part leans, and how far
        ep = Episode(int(seed), profile_name, room_option=room)
        # A leaning part goes against the posed fixture; the fixture keeps clear of its lying spawn too.
        layout = {tp.name: self._start(tp, "lying" if poses[tp.name] == "leaning" else poses[tp.name], xy, yaw,
                                       NOMINAL_POSE) for tp, xy, yaw in draws}
        shapes = {name: start.shape(self._parts[name].part) for name, start in layout.items()}
        variation.draw_fixture(ep, profile, list(shapes.values()),
                               lambda pose, ss: self._home_clear(pose) and all(self._fixture_clear(s, pose) for s in ss))
        for tp, xy, yaw in draws:
            if poses[tp.name] == "leaning":
                layout[tp.name] = self._lean_start(tp, xy, yaw, ep.fixture, layout, pose_rng)
                shapes[tp.name] = layout[tp.name].shape(tp.part)
        self._draw_clutter(ep, profile, layout, shapes)
        scene = {n: (self._parts[n].part, s.xy) for n, s in layout.items()}
        scene.update({n: (self._parts[n].part, xy) for n, (xy, _y) in ep.clutter_pose.items()})
        pool = task.pool
        key = next((k for k in (f"{task.task_id}#{self.variant}", task.task_id) if k in profile.instructions), None)
        if key is not None:
            pool = variation.validate_pool(list(profile.instructions[key]), set(layout),
                                           f"{profile.name} instructions for {key}")
        tray_xy = seats_for(task.fixture_spec or FIXTURE_SPEC, ep.fixture)["tray"].xy
        ep.instruction = variation.choose_instruction(ep, pool, scene, np.array(tray_xy))
        variation.draw_visuals(ep, profile, ("left", "right"))
        return ep, layout

    def _lean_start(self, tp: TaskPart, xy: np.ndarray, yaw: float, fixture: FixturePose, layout: dict,
                    rng: np.random.Generator) -> Start:
        """`tp` leaning against the posed fixture's -y block face, on the table, in reach and 1 cm clear of
        the other parts; lying at its drawn spawn if no place along the face fits (`_LEAN_TRIES`)."""
        others = [s.shape(self._parts[n].part) for n, s in layout.items() if n != tp.name]
        for _ in range(_LEAN_TRIES):
            start = self._start(tp, "leaning", xy, yaw, fixture, rng)
            shape = start.shape(tp.part)
            on_table = all(SPAWN_X[0] + r <= p[0] <= SPAWN_X[1] - r and SPAWN_Y[0] + r <= p[1] <= SPAWN_Y[1] - r
                           and math.hypot(*p) <= SPAWN_REACH - r for a, b, r in shape.caps for p in (a, b))
            if on_table and all(gap(shape, o) >= 0.01 for o in others):
                return start
        return self._start(tp, "lying", xy, yaw, fixture)

    def _home_clear(self, pose: FixturePose) -> bool:
        """The fixture block keeps HOME_CLEARANCE (in plan) from every robot collider that hangs below its
        top + 1 cm with the arm at HOME (the open gripper's pads and fingers): the arm starts every episode
        there, and the block (115 mm) is taller than the home pads are high."""
        spec = self.task.fixture_spec or FIXTURE_SPEC
        (x0, x1), (y0, y1) = ((v * MM for v in spec["block"]["x"]), (v * MM for v in spec["block"]["y"]))
        if self._home_points is None:
            self._home_points = home_keepout(self.model, spec["block"]["top_z"] * MM + 0.01)
        for p in self._home_points:
            u, v = pose.to_local(p)
            if math.hypot(max(x0 - u, u - x1, 0.0), max(y0 - v, v - y1, 0.0)) < HOME_CLEARANCE:
                return False
        return True

    def _fixture_clear(self, shape: Shape, pose: FixturePose) -> bool:
        """The part outline stays _SPAWN_FIXTURE_CLEARANCE off the fixture's footprint."""
        for a, b, r in shape.caps:
            for t in np.linspace(0.0, 1.0, 6):
                if pose.distance(a + t * (b - a)) < r + _SPAWN_FIXTURE_CLEARANCE:
                    return False
        return True

    def _free(self, shape: Shape, pose: FixturePose) -> bool:
        """On the table, within reach and off the fixture."""
        for a, b, r in shape.caps:
            for p in (a, b):
                if not (SPAWN_X[0] + r <= p[0] <= SPAWN_X[1] - r and SPAWN_Y[0] + r <= p[1] <= SPAWN_Y[1] - r):
                    return False
                if math.hypot(*p) > SPAWN_REACH - r:
                    return False
        return self._fixture_clear(shape, pose)

    def _draw_clutter(self, ep: Episode, profile: variation.Profile, layout: dict, shapes: dict) -> None:
        """0..clutter_max parts from the pool, each on free table: apart from the task parts (or, with
        `clutter_touching`, sometimes against one) and never on a goal part's grasp (`grasp_zone`)."""
        # The task's own pool; the eval profile's extra (organizer-private) parts only in eval episodes.
        pool = self.clutter_pool if profile.name == "eval" else self.clutter_pool[:len(self.task.clutter)]
        top = min(profile.clutter_max, len(pool))
        if top <= 0:
            return
        rng = variation.stream(ep.seed, "clutter")
        count = int(rng.integers(0, top + 1))
        chosen = sorted(int(i) for i in rng.choice(len(pool), size=count, replace=False)) if count else []
        anchors = list(shapes.values())
        placed = list(anchors)
        zones = []
        for name in dict.fromkeys(n for g in self.task.goals for n in (g.part, g.on) if n):
            start, part = layout[name], self._parts[name].part
            zones.append(grasp_zone(part, start.xy, start.yaw) if start.pose == "lying"
                         else clear_around(start.shape(part)))
        min_gap = 0.01 if profile.clutter_touching else 0.02
        (x0, x1), (y0, y1) = CLUTTER_REGION
        for index in chosen:
            tp = pool[index]
            for _attempt in range(100):
                yaw = float(rng.uniform(-math.pi, math.pi))
                anchor = None
                if profile.clutter_touching and rng.random() < 0.5:
                    anchor = anchors[int(rng.integers(0, len(anchors)))]
                    angle, want = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0.0, 0.006))
                    centre = 0.5 * (anchor.caps[0][0] + anchor.caps[0][1])
                    direction = np.array([math.cos(angle), math.sin(angle)])
                    xy = None
                    for r in np.arange(0.0, 0.3, 0.002):
                        if gap(shape_of(tp.part, centre + r * direction, yaw), anchor) >= want:
                            xy = centre + r * direction
                            break
                    if xy is None:
                        continue
                else:
                    xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                s = shape_of(tp.part, xy, yaw)
                if not self._free(s, ep.fixture):
                    continue
                if any(gap(s, o) < (0.0 if o is anchor else min_gap) for o in placed):
                    continue
                if any(gap(s, z) < 0.0 for z in zones):
                    continue
                placed.append(s)
                ep.clutter_pose[tp.name] = (xy, yaw)
                break

    def _apply_episode(self, ep: Episode) -> None:
        """Everything of the episode that is not simulator state: seats, clutter rules, visuals, spec."""
        task = self.task
        self.episode = ep
        self.fixture_pose = ep.fixture
        self.seats = seats_for(task.fixture_spec or FIXTURE_SPEC, ep.fixture)
        self.active_clutter = tuple(sorted(ep.clutter_pose, key=lambda n: int(n[len("clutter"):])))
        seats = {g.seat for g in task.goals if g.seat and g.seat != "tray"}
        if any(g.seat == "tray" for g in task.goals):
            seats.add("tray")
        self._clutter_decoys = tuple(Decoy(name, tuple(sorted(seats))) for name in self.active_clutter)
        self._apply_visuals(ep)
        self.spec = replace(self.spec, instruction=ep.instruction)

    def mass(self, name: str) -> float:
        """The part's mass this episode (kg)."""
        return float(self.model.body_mass[self.part_body[name]])

    def _default_visuals(self) -> None:
        """The model's own lights and material colours (no episode variation)."""
        m = self.model
        light_dir, diffuse, specular, ambient, _head, rgba, mat_spec, mat_shine = self._visual_defaults
        m.light_dir[:] = light_dir
        m.light_diffuse[:] = diffuse
        m.light_specular[:] = specular
        m.vis.headlight.ambient[:] = ambient
        m.mat_rgba[:] = rgba
        m.mat_specular[:] = mat_spec
        m.mat_shininess[:] = mat_shine
        self._show_room(rooms.NOMINAL_ROOM)

    def _apply_visuals(self, ep: Episode) -> None:
        m = self.model
        _light_dir, diffuse, specular, ambient, _head, _rgba, _mat_spec, _mat_shine = self._visual_defaults
        self._default_visuals()
        if ep.light_dir is not None:
            m.light_dir[0] = ep.light_dir
            m.light_diffuse[0] = np.clip(diffuse[0] * ep.light_scale, 0.0, 1.0)
            m.light_specular[0] = np.clip(specular[0] * ep.light_scale, 0.0, 1.0)
            m.vis.headlight.ambient[:] = np.clip(ambient * ep.ambient_scale, 0.0, 1.0)

            def tint(material: str, rgb: np.ndarray) -> None:
                idx = m.material(material).id
                m.mat_rgba[idx, :3] = np.clip(m.mat_rgba[idx, :3] * rgb, 0.0, 1.0)

            tint("table_mat", ep.table_rgb)
            tint("mat_fixture_body", ep.fixture_rgb)
            for name in self._parts:
                tint(f"mat_{name}", ep.part_rgb)
        self._apply_cameras(ep)
        self._set_sky(ep.sky_turns)
        self._show_room(ep.room, ep)

    def _show_room(self, name: str, ep: Episode | None = None) -> None:
        """Show room `name` (rooms.RoomSwitch) with its light balance under the episode's light variation;
        a room that is not in the model shows the cube-map backdrop instead."""
        m = self.model
        lit = ep is not None and ep.light_dir is not None
        shown = self._room_switch.show(m, name, ep.light_scale if lit else None)
        ambient = self._visual_defaults[3] * (ep.ambient_scale if lit else 1.0) * self._room_switch.ambient_factor()
        m.vis.headlight.ambient[:] = np.clip(ambient, 0.0, 1.0)
        self._set_sky(self._sky_turns, "exterior" if shown and self._backdrops["exterior"] is not None else "cubemap")

    def _apply_cameras(self, ep: Episode) -> None:
        """The configured camera poses, then the episode's tripod jitter (in each camera's own frame)."""
        m = self.model
        pos, quat, fovy = self._camera_base
        m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = pos, quat, fovy
        for name, (dpos, drot) in ep.camera_offset.items():
            idx = m.camera(name).id
            m.cam_pos[idx] = pos[idx] + dpos
            q = quat[idx].copy()
            mujoco.mju_quatIntegrate(q, np.asarray(drot, float), 1.0)
            m.cam_quat[idx] = q

    def _set_sky(self, turns: int, backdrop: str | None = None) -> None:
        """Turn the background cube map (`backdrop`: the room exterior or the workshop photo; default the
        current one) by `turns` x 90° about z (the four side faces cycle; the top and bottom faces turn in
        place), and upload it to every renderer already made."""
        backdrop = backdrop or self._backdrop
        if turns == self._sky_turns and backdrop == self._backdrop:
            return
        tex, adr, _faces = self._sky
        faces = self._backdrops[backdrop]
        out = faces.copy()
        ring = (0, 5, 1, 4)  # the side faces in azimuth order
        for i, face in enumerate(ring):
            out[ring[(i + turns) % 4]] = faces[face]
        out[2] = np.rot90(faces[2], turns)
        out[3] = np.rot90(faces[3], -turns)
        self.model.tex_data[adr:adr + out.size] = out.ravel()
        for renderer in (self.renderer, *self._sized_renderers.values()):
            if renderer is not None:
                renderer._gl_context.make_current()  # noqa: SLF001 - mujoco.Renderer has no public upload
                mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, tex)  # noqa: SLF001
        self._sky_turns = turns
        self._backdrop = backdrop

    def _initial_state(self, data: mujoco.MjData, ep: Episode, layout: dict) -> None:
        """The episode's start on `data`: arm home, fixture posed, parts down, clutter out, settled."""
        m = self.model
        mujoco.mj_resetData(m, data)  # clutter not in play stays on its shelf (qpos0)
        data.qpos[:NUM_JOINTS] = HOME_Q
        data.qpos[self.grip_qadr] = self.grip_open_q
        data.ctrl[:NUM_JOINTS] = HOME_Q
        data.ctrl[self.grip_act] = self.grip_open_ctrl
        data.mocap_pos[self.fixture_mocap] = [ep.fixture.x, ep.fixture.y, 0.0]
        data.mocap_quat[self.fixture_mocap] = ep.fixture.quat()
        for name, start in layout.items():
            self._set_part(data, name, start.pos, start.quat)
        for name, (xy, yaw) in ep.clutter_pose.items():
            part = self._parts[name].part
            self._set_part(data, name, np.array([xy[0], xy[1], part.lying_radius]), _yaw_quat(yaw))
        mujoco.mj_forward(m, data)
        for _ in range(40):
            mujoco.mj_step(m, data)

    def _capture_episode_goal(self, ep: Episode, layout: dict) -> None:
        """This episode's goal picture: its own start (fixture pose, parts where they lie, clutter,
        lighting and colours) with every goal part seated and the arm parked. Once per episode."""
        key = (ep.key(), self.variant)
        if self._goal_episode == key:
            return
        m, g = self.model, self._goal_data
        self._initial_state(g, ep, layout)
        self._seat_goals(g)
        g.qvel[:] = 0.0
        g.qpos[:NUM_JOINTS] = GOAL_ARM_Q
        g.ctrl[:NUM_JOINTS] = GOAL_ARM_Q
        mujoco.mj_forward(m, g)
        for _ in range(80):
            mujoco.mj_step(m, g)
        self._stage_goals(self.episode_goal, g, ep.seed, eager=self.render)
        self._goal_episode = key

    def reset(self, seed: int, options: dict | None = None) -> RawObs:
        """Start the seed's episode with its selected variation profile."""
        profile = variation.parse_options(options if options is not None else {"profile": self.profile},
                                          internal=True)
        # The goal frame needs a GL context. Capture it on the first reset,
        # which is the thread that keeps rendering. Building it earlier, on the
        # loader thread, makes EGL refuse the warm-up.
        self._ready_goals(self.task_def.variant_index(seed))
        ep, layout = self._draw_episode(seed, profile, (options or {}).get("room"))
        self._apply_episode(ep)
        self._home()
        self._initial_state(self.data, ep, layout)
        self._refresh_scene_kinematics()
        self.step_count = 0
        self.hold = 0
        self.hold_quality, self.hold_quality_limit = 1.0, None
        self.safety_peaks = dict.fromkeys(SAFETY_LIMITS, 0.0)
        self.protective_stop = None
        self._first_met = [-1] * len(self.task.goals)
        self._start_xy = {tp.name: self.part_pose(tp.name)[0][:2] for tp in self.task.parts}
        slots = self._tray_slots()
        self._start_dist = [self._goal_distance(g, *slots.get(g.part, (0, 1))) for g in self.task.goals]
        self.last_eval = None
        self._capture_episode_goal(ep, layout)
        return self._obs()

    def step(self, action: JointDelta) -> StepResult:
        d = self.data
        start = self.target
        values = np.clip(action.values, -JOINT_STEP_CLAMP, JOINT_STEP_CLAMP)
        clamped = bool((values != np.asarray(action.values)).any())
        self.target = np.clip(
            self.target + values, self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
        qd_des = values / CONTROL_DT
        scale, tcp_over, elbow_over = self._speed.scale_step(start, self.target - start)
        if scale < 1.0:
            self.target = start + scale * (self.target - start)
            qd_des = qd_des * scale
        grip = self.grip_closed_ctrl + action.gripper * (self.grip_open_ctrl - self.grip_closed_ctrl)
        self.q_cmd = self._drive(d, self.target, self.q_cmd, qd_des, grip)
        self._refresh_scene_kinematics()
        self.step_count += 1
        obs = self._obs()
        for i, goal in enumerate(self.task.goals):
            if self._first_met[i] < 0 and self._goal_ok(goal):
                self._first_met[i] = self.step_count
        ev = self.evaluate()
        self.last_eval = ev
        self.hold = self.hold + 1 if ev.success_now else 0
        if self.hold == 1:  # a new hold: its quality is the minimum over its samples
            self.hold_quality, self.hold_quality_limit = 1.0, None
        if self.hold and ev.quality < self.hold_quality:
            self.hold_quality, self.hold_quality_limit = ev.quality, ev.quality_limit
        success = self.hold >= HOLD_STEPS and self.protective_stop is None
        result = self._result(obs, success, ev)
        result.limits = {"speed_scaled": {"tcp_speed": int(tcp_over), "elbow_speed": int(elbow_over),
                                          "joint_clamp": int(clamped)},
                         "safety_peaks": dict(self.safety_peaks)}
        if self.protective_stop is not None:  # enforced safety stop: the episode ends here, unsuccessful
            result.done = True
            result.limits["protective_stop"] = dict(self.protective_stop)
        return result

    def _result(self, obs: RawObs, success: bool, ev: Evaluation) -> StepResult:
        limit = self.spec.episode_length if self.max_steps is None else self.max_steps
        done = bool(success or (limit > 0 and self.step_count >= limit))
        if success:
            doneness = 1.0
            speed = speed_bonus(self.step_count, self.spec.episode_length)
            quality, limit = self.hold_quality, self.hold_quality_limit
            score = episode_score(True, speed, quality)
        else:
            doneness = ev.doneness
            speed, quality, limit = 0.0, 0.0, None
            score = episode_score(False, 0.0, 0.0, ev.counted, ev.unmet_progress, len(ev.goals))
        # Energy is no longer scored; the wire field stays 0.0 for older readers.
        return StepResult(obs, done, bool(success), float(score), float(doneness), float(speed), 0.0,
                          float(quality), limit)

    # Older tools and tests ask these of a single-goal task.
    def _in_goal(self) -> bool:
        return self.evaluate().success_now

    def state_hash(self) -> str:
        mujoco.mj_getState(self.model, self.data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return hash_state(self.step_count, self.state, self.q_cmd)

    # -------------------------------------------------------------- rendering

    def set_orbit(self, azimuth: float, elevation: float) -> None:
        """Aim the organizer camera. Azimuth is degrees around z. Elevation is degrees above the table."""
        if self._free_cam is None:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.76, 0.02, 0.08]
            cam.distance = 1.15
            self._free_cam = cam
        self._free_cam.azimuth = float(azimuth) % 360.0
        self._free_cam.elevation = -float(np.clip(elevation, 8.0, 80.0))

    def _goal_metadata(self, seed: int | None) -> dict:
        pose = list(GOAL_ARM_Q)
        # Clear the projected fingertip from the stacked parts in the wrist image.
        if any(goal.on for goal in self.task_def.scenes[self.variant].goals):
            pose[5] -= 0.4
        return {"seed": seed, "variant": self.variant, "wrist_joint_pos": pose}

    def _stage_goals(self, goal_set: GoalSet, data: mujoco.MjData, seed: int | None, *, eager: bool) -> None:
        # Put the wrist at its exact recorded capture pose after settling the parts.
        capture = self._goal_metadata(seed)
        data.qpos[:NUM_JOINTS] = capture["wrist_joint_pos"]
        mujoco.mj_forward(self.model, data)
        goal_set.stage(lambda: {cam.name: self._grab(data, cam.name, (cam.width, cam.height))
                                for cam in CAMERAS}, capture, eager=eager)

    def _lazy_nominal_goals(self) -> dict[str, np.ndarray]:
        # A headless episode may be active. Its model has per-seed materials and camera jitter.
        # Render the nominal scene independently so reading /tasks cannot alter the active episode.
        nominal = TabletopEnv(self.task_def.scenes[0])
        try:
            nominal._capture_goal()
            return nominal.nominal_goal.frames()
        finally:
            nominal.close()

    def close(self) -> None:
        """Free every scene's renderers (their GL contexts). Idempotent; a later render makes new ones."""
        for scene in (self.__dict__, *self._scenes.values()):
            for renderer in (scene.get("renderer"), *scene.get("_sized_renderers", {}).values()):
                if renderer is not None:
                    renderer.close()
            scene["renderer"] = None
            scene["_sized_renderers"] = {}

    def _renderer(self, width: int = 640, height: int = 480) -> mujoco.Renderer:
        if (width, height) == (640, 480):
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 480, 640)
            return self.renderer
        sized = self._sized_renderers.get((width, height))
        if sized is None:
            sized = self._sized_renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        return sized

    def _grab(self, data: mujoco.MjData, camera: str,
              size: tuple[int, int] = (640, 480)) -> np.ndarray:
        renderer = self._renderer(*size)
        if camera == "orbit" and self._free_cam is not None:
            renderer.update_scene(data, camera=self._free_cam)
        else:
            renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return renderer.render().copy()

    def _obs(self) -> RawObs:
        m, d = self.model, self.data
        images: dict[str, np.ndarray] = {}
        self.audience_images = {}
        if self.render:
            images = {cam.name: self._grab(d, cam.name, (cam.width, cam.height)) for cam in self.spec.cameras}
            if self.audience:
                names = ["booth", "side"]
                orbit_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "orbit")
                if orbit_id >= 0:
                    names.append("orbit")
                self.audience_images = {name: self._grab(d, name) for name in names}
        quat = np.zeros(4)  # the TCP in the robot base frame (UR base_link), as the real arm reports it
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.tip])
        return RawObs(
            images=images,
            joint_pos=d.qpos[:NUM_JOINTS].copy(),
            joint_vel=d.qvel[:NUM_JOINTS].copy(),
            gripper=float(np.clip(
                (d.qpos[self.grip_qadr] - self.grip_closed_q) / (self.grip_open_q - self.grip_closed_q), 0.0, 1.0)),
            ee_pos=d.site_xpos[self.tip] - [0.0, 0.0, cell.BASE_Z],
            ee_quat=quat,
        )
