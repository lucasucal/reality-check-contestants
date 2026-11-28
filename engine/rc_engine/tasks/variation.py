"""Per-episode variation of the fastener tasks: everything is drawn from the episode seed.

Same seed and options => the same episode. An episode is reset with options

    {"profile": "public" | "eval"}

(defaults public). The profile sets the ranges; the seed draws within them:

* instruction - one phrasing from the task's pool (`instructions` in the task file).
  A phrasing may carry the condition (`when`) `nearest:<part>:<tray|robot>` (of the
  parts of its kind, it is the nearest to the tray / robot base by at least 3 cm). The
  first phrasing has no conditions. Every part is the same grey PLA, so no phrasing
  names a material or a colour.
* visuals - light direction, intensity and colour temperature, the ambient light,
  the table tint, the fixture's PLA shade, one shade shared by every part, the two
  tripod cameras' poses (the wrist camera is bolted to the arm), and the background
  cube map turned in 90° steps.
* room - one of the furnished rooms around the bench (rooms.py), uniform over the profile's `rooms`
  (all four unless the profile lists them), from a stream of its own; reset option `room` (practice,
  not with "eval") picks one instead.
* fixture pose - x/y and yaw on the table; seats, colliders and goal poses follow.
* clutter - extra kit parts from the task's `clutter` pool, never of a kind that
  makes the instruction ambiguous; one in a goal seat (or the tray, when a goal is
  the tray) is a decoy violation.
* start pose - a part whose spawn lists several poses (lying, standing, leaning, on
  edge; see tabletop.START_POSES) starts in one drawn by weight.

The public profile is `assets/variation/public.json`. The eval profile is organizer
private: it lives with the hidden task packs (`<task-packs>/profiles/eval.json`) and
is loaded with them (`load_eval_profile`); without it, "eval" is refused.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fixture import FIXTURE_ORIGIN, NOMINAL_YAW, FixturePose
from .rooms import NOMINAL_ROOM, ROOM_NAMES

PROFILE_DIR = Path(__file__).resolve().parent.parent / "assets" / "variation"
EVAL_PROFILE_PATH = Path("profiles") / "eval.json"  # inside a mounted task-pack directory
OPTION_KEYS = {"profile", "room"}
PROFILES = ("public", "eval")

NEAREST_MARGIN = 0.03
# Nominal light (tabletop._room): direction (-0.35, 0.15, -1): heading ~157°, ~69° down.
NOMINAL_LIGHT_AZIMUTH = math.degrees(math.atan2(0.15, -0.35))
NOMINAL_LIGHT_ELEVATION = math.degrees(math.atan2(1.0, math.hypot(0.35, 0.15)))

# Independent random streams, so changing one mechanism does not move the others.
_STREAMS = {"instruction": 1, "visual": 2, "fixture": 3, "clutter": 4, "pose": 6, "room": 7}


def stream(seed: int, name: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0x7661, _STREAMS[name]])))


# ------------------------------------------------------------------ profiles

@dataclass(frozen=True)
class Profile:
    name: str
    light_azimuth_deg: float = 0.0  # ± about the nominal azimuth
    light_elevation_deg: tuple[float, float] = (NOMINAL_LIGHT_ELEVATION, NOMINAL_LIGHT_ELEVATION)
    light_intensity: tuple[float, float] = (1.0, 1.0)
    light_kelvin: tuple[float, float] = (6500.0, 6500.0)
    ambient: tuple[float, float] = (1.0, 1.0)
    table_tint: float = 0.0  # ± per channel, multiplicative
    fixture_shade: float = 0.0  # ± brightness, plus a third of it per channel
    part_shade: float = 0.0  # one shade for every part
    camera_pos_m: float = 0.0  # ± per axis, tripod cameras
    camera_rot_deg: float = 0.0  # ± about each axis
    sky_rotation: bool = False
    rooms: tuple[str, ...] = ROOM_NAMES  # drawn uniformly per episode (draw_room)
    fixture_xy_m: float = 0.0  # ± per axis
    fixture_yaw_deg: float = 0.0  # ±
    clutter_max: int = 0
    clutter_touching: bool = False  # some clutter may lie against a task part (never on its grasp)
    instructions: dict[str, tuple] = field(default_factory=dict)  # task_id -> novel phrasings (eval)
    clutter_parts: dict[str, tuple] = field(default_factory=dict)  # task_id -> extra clutter pool (eval)
    part_manifest: Path | None = None  # kit for clutter_parts (the pack's manifest)

    @classmethod
    def from_json(cls, doc: dict, root: Path | None = None) -> "Profile":
        # Older mounted profiles may contain a nudge section; it has no effect.
        known = {"profile", "light", "table_tint", "fixture_shade", "part_shade", "camera_jitter", "sky_rotation",
                 "rooms", "fixture_pose", "clutter", "nudge", "instructions", "clutter_parts", "part_manifest", "notes"}
        extra = set(doc) - known
        if extra:
            raise ValueError(f"unknown profile fields {sorted(extra)}")
        light = doc.get("light", {})
        cam = doc.get("camera_jitter", {})
        pose = doc.get("fixture_pose", {})
        clutter = doc.get("clutter", {})

        def pair(v, default):
            return tuple(float(x) for x in v) if v is not None else default

        manifest = None
        if doc.get("part_manifest"):
            if root is None:
                raise ValueError("part_manifest needs the profile's pack directory")
            manifest = (root / doc["part_manifest"]).resolve()
            if not manifest.is_relative_to(root.resolve()) or not manifest.is_file():
                raise ValueError(f"profile part_manifest {doc['part_manifest']!r} is missing or outside the pack")
        profile = cls(
            name=str(doc["profile"]),
            light_azimuth_deg=float(light.get("azimuth_deg", 0.0)),
            light_elevation_deg=pair(light.get("elevation_deg"), (NOMINAL_LIGHT_ELEVATION,) * 2),
            light_intensity=pair(light.get("intensity"), (1.0, 1.0)),
            light_kelvin=pair(light.get("kelvin"), (6500.0, 6500.0)),
            ambient=pair(light.get("ambient"), (1.0, 1.0)),
            table_tint=float(doc.get("table_tint", 0.0)),
            fixture_shade=float(doc.get("fixture_shade", 0.0)),
            part_shade=float(doc.get("part_shade", 0.0)),
            camera_pos_m=float(cam.get("pos_m", 0.0)),
            camera_rot_deg=float(cam.get("rot_deg", 0.0)),
            sky_rotation=bool(doc.get("sky_rotation", False)),
            rooms=tuple(doc.get("rooms", ROOM_NAMES)),
            fixture_xy_m=float(pose.get("xy_m", 0.0)),
            fixture_yaw_deg=float(pose.get("yaw_deg", 0.0)),
            clutter_max=int(clutter.get("max", 0)),
            clutter_touching=bool(clutter.get("touching", False)),
            instructions={k: tuple(v) for k, v in (doc.get("instructions") or {}).items()},
            clutter_parts={k: tuple(v) for k, v in (doc.get("clutter_parts") or {}).items()},
            part_manifest=manifest,
        )
        if (not profile.rooms or len(set(profile.rooms)) != len(profile.rooms)
                or not set(profile.rooms) <= set(ROOM_NAMES)):
            raise ValueError(f"rooms must list distinct rooms from {list(ROOM_NAMES)}")
        if not 0 <= profile.clutter_max <= 4:
            raise ValueError("clutter.max must be 0-4")
        if profile.fixture_xy_m > 0.06 or profile.fixture_yaw_deg > 35 or profile.camera_pos_m > 0.05:
            raise ValueError("profile ranges are out of bounds (fixture ±6 cm / ±35°, camera ±5 cm)")
        return profile


NOMINAL = Profile("nominal", rooms=(NOMINAL_ROOM,))  # no variation at all: tests and tools that need the documented scene
PUBLIC = Profile.from_json(json.loads((PROFILE_DIR / "public.json").read_text()))
_EVAL: Profile | None = None


def load_eval_profile(pack_dir: str | Path) -> Profile | None:
    """Load `<pack_dir>/profiles/eval.json` if it exists (organizer engines); returns it."""
    global _EVAL
    root = Path(pack_dir)
    path = root / EVAL_PROFILE_PATH
    if not path.is_file():
        return None
    profile = Profile.from_json(json.loads(path.read_text()), root)
    if profile.name != "eval":
        raise ValueError(f"{path}: profile must be named 'eval'")
    _EVAL = profile
    return profile


def eval_profile() -> Profile | None:
    return _EVAL


def profile_named(name: str) -> Profile:
    if name == "public":
        return PUBLIC
    if name == "nominal":
        return NOMINAL
    if name == "eval":
        if _EVAL is None:
            raise ValueError("eval profile not mounted (needs <task-packs>/profiles/eval.json)")
        return _EVAL
    raise ValueError(f"profile must be one of {', '.join(PROFILES)}, got {name!r}")


def parse_options(options: object, *, internal: bool = False) -> str:
    """Validate reset options and return the profile. `internal` also allows the 'nominal' profile."""
    if options is None:
        return "public"
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    extra = set(options) - OPTION_KEYS
    if extra:
        raise ValueError(f"unknown options {sorted(extra)}; allowed: {sorted(OPTION_KEYS)}")
    profile = options.get("profile", "public")
    allowed = PROFILES + (("nominal",) if internal else ())
    if profile not in allowed:
        raise ValueError(f"options.profile must be one of {', '.join(PROFILES)}, got {profile!r}")
    profile_named(profile)  # eval must be mounted
    room = options.get("room")
    if room is not None and (room not in ROOM_NAMES or profile == "eval"):
        raise ValueError(f"options.room must be one of {', '.join(ROOM_NAMES)}, and not with the eval profile")
    return profile


# ------------------------------------------------------------------ geometry of lying parts

@dataclass(frozen=True)
class Shape:
    """Top-view outline of a lying part: capsules (segment a-b, radius)."""

    caps: tuple[tuple[np.ndarray, np.ndarray, float], ...]


def _seg_dist(p1, q1, p2, q2) -> float:
    """Distance between two 2-D segments."""
    def point_seg(p, a, b):
        ab = b - a
        t = 0.0 if float(ab @ ab) < 1e-12 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    d1, d2 = q1 - p1, q2 - p2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) > 1e-12:
        r = p2 - p1
        t = (r[0] * d2[1] - r[1] * d2[0]) / den
        u = (r[0] * d1[1] - r[1] * d1[0]) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(point_seg(p1, p2, q2), point_seg(q1, p2, q2), point_seg(p2, p1, q1), point_seg(q2, p1, q1))


def gap(a: Shape, b: Shape) -> float:
    return min(_seg_dist(p1, q1, p2, q2) - r1 - r2 for p1, q1, r1 in a.caps for p2, q2, r2 in b.caps)


def shape_of(part, xy, yaw: float) -> Shape:
    """`part` is a tabletop.Part lying at xy with body yaw (bolt head along yaw)."""
    xy = np.asarray(xy, float)[:2]
    if part.shape == "bolt":
        d = np.array([math.cos(yaw), math.sin(yaw)])
        head = xy + d * part.head_local_x
        return Shape(((xy + d * part.tip_x, xy + d * part.under_head_x, 0.5 * part.diameter),
                      (head, head, part.across_flats / math.sqrt(3.0))))
    return Shape(((xy, xy, part.footprint),))


def shape_at(part, pos, rot) -> Shape:
    """Top-view outline of `part` at any body pose (world pos, 3x3 rot): lying, standing, leaning, on edge."""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    xy = pos[:2]
    if part.shape == "bolt":
        axis = rot[:2, 0]
        head = xy + axis * part.head_local_x
        return Shape(((xy + axis * part.tip_x, xy + axis * part.under_head_x, 0.5 * part.diameter),
                      (head, head, part.across_flats / math.sqrt(3.0))))
    up = abs(float(rot[2, 2]))  # the part's axis against vertical: 1 lying flat, 0 on its edge
    if up > 0.9:
        return Shape(((xy, xy, part.footprint),))
    # On its edge: the face is a disc of radius `footprint` seen side on, a capsule across the axis.
    across = np.array([-rot[1, 2], rot[0, 2]])
    across /= max(float(np.linalg.norm(across)), 1e-9)
    half = 0.5 * part.thickness
    reach = max(part.footprint - half, 0.0)
    return Shape(((xy - reach * across, xy + reach * across, half),))


GRASP_CLEARANCE = 0.035  # beyond half the grasp width: the open pads (+17.5 mm) and their 7.6 mm thickness, + margin


def grasp_zone(part, xy, yaw: float) -> Shape:
    """Where the pads come down to take a goal part: a bolt's shank from near its centre to the head,
    a nut or washer all round. Clutter keeps out of it."""
    xy = np.asarray(xy, float)[:2]
    if part.shape == "bolt":
        d = np.array([math.cos(yaw), math.sin(yaw)])
        return Shape(((xy - 0.02 * d, xy + d * part.under_head_x, 0.5 * part.diameter + GRASP_CLEARANCE),))
    width = part.across_flats if part.shape == "nut" else part.outer_diameter
    return Shape(((xy, xy, 0.5 * width + GRASP_CLEARANCE),))


def clear_around(shape: Shape) -> Shape:
    """A part that does not lie flat (standing, leaning, on edge) is taken from any side: its whole
    outline grown by GRASP_CLEARANCE."""
    return Shape(tuple((a, b, r + GRASP_CLEARANCE) for a, b, r in shape.caps))


# ------------------------------------------------------------------ episode draw

@dataclass
class Episode:
    seed: int
    profile: str
    instruction: str = ""
    fixture: FixturePose = field(default_factory=FixturePose)
    light_dir: np.ndarray | None = None
    light_scale: np.ndarray | None = None  # rgb multiplier of the light's diffuse and specular
    ambient_scale: float = 1.0
    table_rgb: np.ndarray | None = None
    fixture_rgb: np.ndarray | None = None
    part_rgb: np.ndarray | None = None  # rgb multiplier shared by every part
    clutter_pose: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)
    camera_offset: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # name -> (dpos, rotvec)
    sky_turns: int = 0
    room: str = ""
    room_option: str | None = None  # reset option `room`: shown instead of the drawn room

    def key(self) -> tuple:
        return (self.seed, self.profile, self.room_option)


def kelvin_rgb(kelvin: float) -> np.ndarray:
    """Relative RGB of a black body (Tanner Helland's fit), normalised so 6500 K is white."""
    def raw(k: float) -> np.ndarray:
        t = k / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
        return np.clip(np.array([r, g, b]), 0.0, 255.0)
    return raw(kelvin) / raw(6500.0)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def draw_visuals(ep: Episode, profile: Profile, cameras: tuple[str, ...]) -> None:
    rng = stream(ep.seed, "visual")
    az = math.radians(NOMINAL_LIGHT_AZIMUTH + _u(rng, -profile.light_azimuth_deg, profile.light_azimuth_deg))
    el = math.radians(_u(rng, *profile.light_elevation_deg))
    # The light travels along (azimuth, -elevation): its horizontal heading and its angle down to the table.
    ep.light_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), -math.sin(el)])
    ep.light_scale = _u(rng, *profile.light_intensity) * kelvin_rgb(_u(rng, *profile.light_kelvin))
    ep.ambient_scale = _u(rng, *profile.ambient)
    t = profile.table_tint
    ep.table_rgb = 1.0 + rng.uniform(-t, t, 3) if t else np.ones(3)
    f = profile.fixture_shade
    ep.fixture_rgb = (1.0 + _u(rng, -f, f)) + rng.uniform(-f / 3, f / 3, 3) if f else np.ones(3)
    p = profile.part_shade
    ep.part_rgb = (1.0 + _u(rng, -p, p)) + rng.uniform(-p / 3, p / 3, 3) if p else np.ones(3)
    for cam in cameras:
        dpos = rng.uniform(-profile.camera_pos_m, profile.camera_pos_m, 3) if profile.camera_pos_m else np.zeros(3)
        drot = (np.radians(rng.uniform(-profile.camera_rot_deg, profile.camera_rot_deg, 3))
                if profile.camera_rot_deg else np.zeros(3))
        ep.camera_offset[cam] = (dpos, drot)
    ep.sky_turns = int(rng.integers(0, 4)) if profile.sky_rotation else 0
    ep.room = ep.room_option or draw_room(ep.seed, profile)


def draw_room(seed: int, profile: Profile) -> str:
    """The episode's room: uniform over the profile's rooms, a function of the seed alone (its own stream)."""
    rooms = profile.rooms
    return rooms[int(stream(seed, "room").integers(0, len(rooms)))] if len(rooms) > 1 else rooms[0]


def draw_fixture(ep: Episode, profile: Profile, shapes: list[Shape], clear_of, tries: int = 60) -> None:
    """Fixture pose: uniform in the profile's ranges, redrawn until every part lies clear of it
    (`clear_of(pose, shapes)`); the nominal pose if none of `tries` draws fits."""
    if not profile.fixture_xy_m and not profile.fixture_yaw_deg:
        return
    rng = stream(ep.seed, "fixture")
    for _ in range(tries):
        dx, dy = rng.uniform(-profile.fixture_xy_m, profile.fixture_xy_m, 2)
        yaw = NOMINAL_YAW + math.radians(_u(rng, -profile.fixture_yaw_deg, profile.fixture_yaw_deg))
        pose = FixturePose(FIXTURE_ORIGIN[0] + float(dx), FIXTURE_ORIGIN[1] + float(dy), yaw)
        if clear_of(pose, shapes):
            ep.fixture = pose
            return


def check_conditions(when: tuple[str, ...], scene: dict[str, tuple], tray_xy: np.ndarray) -> bool:
    """`scene`: active part name -> (part, xy). See the module docstring for the conditions."""
    for cond in when:
        kind, name, *rest = cond.split(":")
        value = rest[0] if rest else ""
        if name not in scene or kind != "nearest":
            return False
        part, xy = scene[name]
        ref = tray_xy if value == "tray" else np.zeros(2)
        mine = float(np.linalg.norm(np.asarray(xy) - ref))
        rivals = [float(np.linalg.norm(np.asarray(oxy) - ref)) for n, (op, oxy) in scene.items()
                  if n != name and op.shape == part.shape]
        if any(r < mine + NEAREST_MARGIN for r in rivals):
            return False
    return True


def choose_instruction(ep: Episode, pool: tuple, scene: dict[str, tuple], tray_xy: np.ndarray) -> str:
    """One phrasing from `pool` (entries: text, or {text, when}) whose conditions hold in this scene."""
    rng = stream(ep.seed, "instruction")
    order = rng.permutation(len(pool))
    for i in order:
        entry = pool[int(i)]
        text, when = (entry, ()) if isinstance(entry, str) else (entry["text"], tuple(entry.get("when", ())))
        if check_conditions(when, scene, tray_xy):
            return text
    first = pool[0]
    return first if isinstance(first, str) else first["text"]


def validate_pool(pool: object, part_names: set[str], where: str) -> tuple:
    """Instruction pool: 1-12 nonempty phrasings, strings or {text, when}; the first unconditional."""
    if not isinstance(pool, list) or not 1 <= len(pool) <= 12:
        raise ValueError(f"{where} must list 1 to 12 phrasings")
    out = []
    texts = set()
    for i, entry in enumerate(pool):
        if isinstance(entry, str):
            text, when = entry, []
        elif isinstance(entry, dict) and set(entry) <= {"text", "when"} and "text" in entry:
            text, when = entry["text"], entry.get("when", [])
        else:
            raise ValueError(f"{where}[{i}] must be a string or {{text, when}}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}[{i}] text must be a nonempty string")
        if i == 0 and when:
            raise ValueError(f"{where}[0] must have no conditions (it is the fallback)")
        if not isinstance(when, list):
            raise ValueError(f"{where}[{i}].when must be a list")
        for cond in when:
            bits = cond.split(":") if isinstance(cond, str) else []
            if not (len(bits) == 3 and bits[0] == "nearest" and bits[1] in part_names
                    and bits[2] in ("tray", "robot")):
                raise ValueError(f"{where}[{i}]: bad condition {cond!r}")
        if "{" in text or "}" in text:
            raise ValueError(f"{where}[{i}]: phrasings take no placeholders, got {text!r}")
        if text.strip() in texts:
            raise ValueError(f"{where}: duplicate phrasing {text!r}")
        texts.add(text.strip())
        out.append(text.strip() if not when else {"text": text.strip(), "when": list(when)})
    return tuple(out)
