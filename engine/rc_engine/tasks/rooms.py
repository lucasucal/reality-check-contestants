"""The rooms around the robot cell.

Four rooms (`ROOM_NAMES`) surround the cell: three from the SimReady Hardware dataset
(assets/rooms/VENDORED.md), standing on the real cell's floor (cell.FLOOR_Z; their placements keep the
bench feet at -0.75 m, so they are lowered), and `real_cell`, the team's scans of the venue around the
real-arm cell, placed by cell.json `scan_to_robot` (its frame is the scans'). They are visual only: every
room geom has contype = conaffinity = 0 and no mass, so physics,
state and the experts are the same in every room; the fallen-part floor plane and the bench stay
tabletop.py's own. All four rooms are compiled into every scene and one is shown at a time by moving
the others' geoms to a hidden geom group (`HIDDEN_GROUP`), so a reset never recompiles and a hidden
room costs nothing to render. Each room also brings

* a fill light: the scene's window rect lights as one directional light without shadows (power,
  colour and direction from room.json `lighting`; `fill_diffuse`), on only while its room is shown;
* an ambient level: the headlight's ambient times `ambient_factor` (the dome's intensity and any
  ceiling light);
* the exterior: the dataset's dome HDRI as the background cube map (`exterior_sky.png`), seen through
  the windows, turned by the episode's sky turns like the old backdrop.

The per-episode lighting variation (variation.draw_visuals) applies on top: the main light as before,
its colour and intensity scale on the fill light too, its ambient scale times the room's factor.

The episode's room is `Episode.room`, drawn uniformly from the profile's `rooms` by the seed
(variation.draw_room). If a room's files are missing or broken it is left out of the build (a
warning); an episode drawn into it shows the bench against the old workshop cube map instead (the
fallback), as does every episode if the room build fails to compile.
"""
from __future__ import annotations

import functools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .. import cell

ROOM_DIR = Path(__file__).resolve().parent.parent / "assets" / "rooms"
ROOM_NAMES: tuple[str, ...] = ("workshop", "office", "industrial_bay", "real_cell")
NOMINAL_ROOM = "workshop"  # the documented scene (profile "nominal") and the task list's goal pictures
SHOWN_GROUP, HIDDEN_GROUP = 0, 5  # renderers show geom groups 0-2 only
DATASET_FLOOR_Z = -0.75  # the dataset rooms' floor in their placement (convert_rooms.py: the old bench feet)
PREFIX = "room_"
SKY_FILE = "exterior_sky.png"
# The cell's statistic without the rooms (tabletop scene, measured): MuJoCo sizes the directional
# light's shadow map and the cameras' clip planes from it. Pinned so the rooms (10-15 m across) do not
# blur the bench's shadows.
CELL_EXTENT = 1.98
CELL_CENTER = (0.48, 0.0, -0.043)
# room.json `lighting` -> MuJoCo. Powers are USD intensity x 2^exposure x area; the office window fill
# (7000 x 5.23 m x 2.71 m) is the reference.
FILL_REF = 100_000.0
FILL_DIFFUSE = 0.12  # the reference fill's diffuse: enough to lift the shadow side, not to wash out pale floors
FILL_DOWN = 0.5  # daylight through a window comes down: this much -z added to the fill direction
DOME_REF = 2400.0
# Room surfaces' albedo scale. MuJoCo adds its lights without exposure control, so pale floors and walls
# (the bay's polished concrete, mean sRGB 164) under the main light, the headlight and the fill clip to
# white; this keeps their texture. The bench, the robot and the parts are not scaled.
ROOM_ALBEDO = 0.75


@dataclass(frozen=True)
class Room:
    name: str
    root: Path
    manifest: dict

    @property
    def fill_diffuse(self) -> np.ndarray:
        fill = self.manifest.get("lighting", {}).get("fill")
        if not fill:
            return np.zeros(3)
        return FILL_DIFFUSE * math.sqrt(min(float(fill["power"]) / FILL_REF, 2.0)) * np.asarray(fill["rgb"], float)

    @property
    def fill_dir(self) -> np.ndarray:
        fill = self.manifest.get("lighting", {}).get("fill") or {"dir": [0.0, 0.0, -1.0]}
        d = np.asarray(fill["dir"], float) + [0.0, 0.0, -FILL_DOWN]
        return d / np.linalg.norm(d)

    @property
    def placement(self) -> tuple[np.ndarray, np.ndarray]:
        """(rotation, position) of the room's frame in the world: the scans' frame for real_cell, else the
        dataset rooms' engine frame lowered from their bench feet (-0.75 m) to the real cell's floor."""
        if self.manifest.get("placement", {}).get("frame") == "scan":
            return cell.scan_to_world()
        floor = float(self.manifest.get("placement", {}).get("floor_z", DATASET_FLOOR_Z))
        return np.eye(3), np.array([0.0, 0.0, cell.FLOOR_Z - floor])

    @property
    def ambient_factor(self) -> float:
        light = self.manifest.get("lighting", {})
        dome = math.sqrt(max(float(light.get("dome", DOME_REF)), 0.0) / DOME_REF)
        ceiling = 1.0 + 0.3 * min(float(light.get("ceiling", 0.0)) / FILL_REF, 1.0)
        return float(np.clip(dome * ceiling, 0.85, 1.15))


def _quat(rot: np.ndarray) -> list[float]:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot).ravel())
    return q.tolist()


def load_room(name: str, root: Path | None = None) -> Room:
    """Read and check a room's manifest and files (ValueError/OSError if anything is missing)."""
    folder = (root or ROOM_DIR) / name
    manifest = json.loads((folder / "room.json").read_text())
    if manifest.get("room") != name or not manifest.get("materials"):
        raise ValueError(f"{folder}/room.json is not the {name} room")
    for mat in manifest["materials"]:
        for key in ("mesh", "texture"):
            if key in mat and not (folder / mat[key]).is_file():
                raise ValueError(f"{name}: missing {mat[key]}")
    return Room(name, folder, manifest)


def _add_visuals(spec: mujoco.MjSpec, prefix: str, folder: Path, materials: list, group: int,
                 albedo: float = 1.0, placement: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """One mesh geom per material, `placement` (rotation, position) in the world; no bodies, so the state
    (and a replay snapshot) does not depend on which rooms loaded."""
    rot, pos = placement if placement is not None else (np.eye(3), np.zeros(3))
    textures: dict[str, str] = {}
    for mat in materials:
        name = f"{prefix}{mat['name']}"
        m = spec.add_material(name=name)
        m.rgba = [*(albedo * float(c) for c in mat["rgba"][:3]), float(mat["rgba"][3])]
        m.specular = float(mat["specular"])
        m.shininess = float(mat["shininess"])
        m.reflectance = 0.0
        if "texture" in mat:
            tex = textures.get(mat["texture"])
            if tex is None:
                tex = textures[mat["texture"]] = f"{prefix}tex_{Path(mat['texture']).stem}"
                spec.add_texture(name=tex, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(folder / mat["texture"]))
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
            m.texrepeat = [1.0, 1.0]
            m.texuniform = False
        mesh = spec.add_mesh(name=name, file=str(folder / mat["mesh"]))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL  # open surfaces (floors, panels)
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, material=name,
                                pos=pos.tolist(), quat=_quat(rot), contype=0, conaffinity=0, density=0, group=group)


def add_rooms(spec: mujoco.MjSpec, names: tuple[str, ...] = ROOM_NAMES, root: Path | None = None) -> tuple[str, ...]:
    """Add each room (hidden) with its fill light (off); returns the rooms added. A room that does not
    load is skipped with a warning."""
    added = []
    for name in names:
        try:
            room = load_room(name, root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"room {name!r} not loaded ({exc}); its episodes use the cube-map backdrop", stacklevel=2)
            continue
        prefix = f"{PREFIX}{name}_"
        rot, pos = room.placement
        _add_visuals(spec, prefix, room.root, room.manifest["materials"], HIDDEN_GROUP, ROOM_ALBEDO, (rot, pos))
        light = spec.worldbody.add_light(name=f"{prefix}fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                         pos=[0.6, 0.0, 2.0], dir=list(rot @ room.fill_dir), castshadow=False,
                                         diffuse=list(room.fill_diffuse), specular=[0.0, 0.0, 0.0],
                                         ambient=[0.0, 0.0, 0.0])
        light.active = False
        added.append(name)
    if added:
        spec.stat.extent = CELL_EXTENT
        spec.stat.center = list(CELL_CENTER)
    return tuple(added)


@functools.lru_cache(maxsize=4)
def exterior_faces(width: int, root: Path | None = None) -> np.ndarray | None:
    """The exterior cube map as (6, width, width, 3) uint8, decoded by MuJoCo like the scene's own sky."""
    path = (root or ROOM_DIR) / SKY_FILE
    if not path.is_file():
        return None
    spec = mujoco.MjSpec()
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, file=str(path), gridsize=[2, 3],
                     gridlayout="RLUDFB......")
    try:
        m = spec.compile()
    except ValueError as exc:
        warnings.warn(f"exterior cube map not loaded ({exc}); using the workshop backdrop", stacklevel=2)
        return None
    w = int(m.tex_width[0])
    faces = m.tex_data[: 6 * w * w * 3].reshape(6, w, w, 3)
    if w != width:
        from PIL import Image  # noqa: PLC0415
        faces = np.stack([np.asarray(Image.fromarray(f).resize((width, width), Image.BILINEAR)) for f in faces])
    faces = np.ascontiguousarray(faces)
    faces.setflags(write=False)
    return faces


class RoomSwitch:
    """Which room a compiled model shows: geom groups, fill lights and the backdrop."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.geoms: dict[str, np.ndarray] = {}
        self.lights: dict[str, int] = {}
        self.rooms: dict[str, Room] = {}
        for name in ROOM_NAMES:
            prefix = f"{PREFIX}{name}_"
            ids = [g for g in range(model.ngeom) if model.geom(g).name.startswith(prefix)]
            if ids:
                self.geoms[name] = np.array(ids)
                self.lights[name] = model.light(f"{prefix}fill").id
                self.rooms[name] = load_room(name)
        self.shown: str | None = None

    def show(self, model: mujoco.MjModel, name: str | None, light_scale: np.ndarray | None = None) -> bool:
        """Show room `name` (None, or a room not in the model: none, the fallback); returns whether a
        room is shown. `light_scale` (the episode's rgb light scale) tints the fill light."""
        for room, ids in self.geoms.items():
            model.geom_group[ids] = SHOWN_GROUP if room == name else HIDDEN_GROUP
            lid = self.lights[room]
            model.light_active[lid] = room == name
            if room == name:
                scale = np.ones(3) if light_scale is None else np.asarray(light_scale, float)
                model.light_diffuse[lid] = np.clip(self.rooms[room].fill_diffuse * scale, 0.0, 1.0)
        self.shown = name if name in self.geoms else None
        return self.shown is not None

    def ambient_factor(self) -> float:
        return self.rooms[self.shown].ambient_factor if self.shown else 1.0
