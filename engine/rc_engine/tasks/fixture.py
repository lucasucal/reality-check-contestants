"""Printed fixture v3: a low tray, three vertical holes and three hex pockets.

`assets/fixture/fixture_v3.json` is the source of truth (millimetres, fixture
frame: origin at the footprint centre on the table, z up; designed with +x away from
the robot). This module places that frame in the world, names its seats, and builds
the colliders. The printed STL is visual only.

Colliders are convex pieces, so the holes, pockets and tray are real:

* the tray floor slab and its three walls are boxes;
* the raised block is boxes, layered by feature floor depth, with a square cut
  around every feature that is open in that layer;
* each cut is filled back to the feature's profile with convex prisms, one per
  polygon edge (a 24-gon for a round hole, the hexagon for a pocket). The 45°
  mouth chamfer of each hole (8 mm) is modelled as a 24-sided cone and each
  pocket's 4 mm lead-in as a hexagonal frustum, so a part released a little off
  the axis slides in; only the block's 2 mm edge chamfer is visual only.

Pose: the fixture is one mocap body (`fixture`), so an episode can move it on the
table (`FixturePose`: x, y and yaw; see `variation.py`). Its colliders, the seats
(`seats_for(spec, pose)`) and every seat test follow the pose; the episode's pose
is in the simulator state (mocap), so replay snapshots carry it.

Contact: every fixture collider is printed PLA (`FIXTURE_FRICTION`, sliding
0.4) with contact `FIXTURE_PRIORITY` above the parts', so MuJoCo uses the
fixture's friction for part-fixture contacts instead of the larger of the pair
(the parts' 1.2). The finger pads have the same priority (1): pad-fixture
contacts keep MuJoCo's max rule, and pad-part contacts are unaffected.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..cell import FIXTURE_NOMINAL, FIXTURE_RGBA

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fixture"
SPEC_PATH = FIXTURE_DIR / "fixture_v3.json"
BODY_MESH = FIXTURE_DIR / "fixture_body.stl"
TEXT_MESH = FIXTURE_DIR / "fixture_text.stl"

FIXTURE_SPEC: dict = json.loads(SPEC_PATH.read_text())
MM = 0.001
# The nominal pose: the real fixture where it stands in the real cell (cell.json `fixture.nominal`,
# measured in the scans), on the table top. Its yaw turns the tray toward +y and the block's -y face
# (lettering) toward the robot base.
FIXTURE_ORIGIN = (float(FIXTURE_NOMINAL["xy"][0]), float(FIXTURE_NOMINAL["xy"][1]), 0.0)
NOMINAL_YAW = math.radians(float(FIXTURE_NOMINAL["yaw_deg"]))
HALF_FOOTPRINT = (0.5 * FIXTURE_SPEC["footprint"][0] * MM, 0.5 * FIXTURE_SPEC["footprint"][1] * MM)

BODY_RGBA = FIXTURE_RGBA  # the real print's cyan PLA (photos)
TEXT_RGBA = (0.93, 0.93, 0.90, 1.0)  # REALITY CHECK inlay
# Printed PLA against the printed PLA parts (or metal, nylon): sliding 0.3-0.5 (0.4 here). The parts
# carry 1.2 for the table and the finger pads; without the priority MuJoCo would take the larger
# coefficient of the pair and nothing could slide down a 45° chamfer (tan 45° = 1).
FIXTURE_FRICTION = (0.4, 0.02, 0.001)
# Above the parts (0), equal to the finger pads (1): part-fixture contacts use the fixture's friction,
# condim and solver parameters; pad-fixture contacts mix as before.
FIXTURE_PRIORITY = 1
FIXTURE_CONDIM = 4  # the parts' condim, so a part on the fixture keeps its torsional friction
HOLE_SIDES = 24
_FEATURE_MARGIN_MM = 4.0  # solid wall between a feature's mouth and its square cut


@dataclass(frozen=True)
class FixturePose:
    """Where the fixture origin sits on the table (world metres) and its yaw about z (radians)."""

    x: float = FIXTURE_ORIGIN[0]
    y: float = FIXTURE_ORIGIN[1]
    yaw: float = NOMINAL_YAW

    def to_world(self, local) -> np.ndarray:
        """Fixture-frame xy (metres, origin at the fixture origin) -> world xy."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx, ly = float(local[0]), float(local[1])
        return np.array([self.x + c * lx - s * ly, self.y + s * lx + c * ly])

    def to_local(self, world) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = float(world[0]) - self.x, float(world[1]) - self.y
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def corners(self, margin: float = 0.0) -> np.ndarray:
        """World xy of the footprint's four corners, grown by `margin`."""
        hx, hy = HALF_FOOTPRINT[0] + margin, HALF_FOOTPRINT[1] + margin
        return np.array([self.to_world(p) for p in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])

    def distance(self, world) -> float:
        """Distance from a world xy point to the footprint rectangle (0 inside)."""
        lx, ly = self.to_local(world)
        return float(math.hypot(max(abs(lx) - HALF_FOOTPRINT[0], 0.0), max(abs(ly) - HALF_FOOTPRINT[1], 0.0)))

    def quat(self) -> np.ndarray:
        return np.array([math.cos(0.5 * self.yaw), 0.0, 0.0, math.sin(0.5 * self.yaw)])


NOMINAL_POSE = FixturePose()


@dataclass(frozen=True)
class Seat:
    """A place a part can be put. World metres, for the fixture `pose` the seat was made for.

    `x_range`/`y_range` (tray) are the inner rectangle's world bounds when the fixture is not
    rotated; for a posed fixture test points with `local()` / `in_tray()` against `half`.
    """

    seat_id: str
    kind: str  # "tray", "hole_vertical", "hex_pocket"
    xy: tuple[float, float]
    floor_z: float
    top_z: float
    diameter: float = 0.0  # hole bore
    chamfer: float = 0.0  # 45° mouth chamfer (hole) or lead-in (hex pocket)
    across_flats: float = 0.0  # hex pocket
    flat_normal_yaw: float = 0.0  # hex pocket: yaw of one flat's outward normal
    x_range: tuple[float, float] = (0.0, 0.0)  # tray inner
    y_range: tuple[float, float] = (0.0, 0.0)
    intended_part: str = ""
    pose: FixturePose = NOMINAL_POSE
    half: tuple[float, float] = (0.0, 0.0)  # tray inner half extents along the fixture's x and y

    @property
    def yaw(self) -> float:
        return self.pose.yaw

    def local(self, world) -> np.ndarray:
        """A world xy point in this seat's frame: origin at the seat centre, axes along the fixture's."""
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        dx, dy = float(world[0]) - self.xy[0], float(world[1]) - self.xy[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def world(self, local) -> np.ndarray:
        """Seat-frame xy -> world xy (the inverse of `local`)."""
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        return np.array([self.xy[0] + c * local[0] - s * local[1], self.xy[1] + s * local[0] + c * local[1]])

    def in_tray(self, world, margin: float = 0.0) -> bool:
        """The xy point lies inside the tray's inner rectangle (shrunk by `margin`)."""
        u, v = self.local(world)
        return abs(u) <= self.half[0] - margin and abs(v) <= self.half[1] - margin

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def apothem(self) -> float:
        return 0.5 * self.across_flats


def seats_for(spec: dict, pose: FixturePose = NOMINAL_POSE) -> dict[str, Seat]:
    """The fixture's seats in world metres, for the fixture standing at `pose`."""
    oz = FIXTURE_ORIGIN[2]
    top = oz + spec["block"]["top_z"] * MM

    def world_xy(local_mm: list[float]) -> tuple[float, float]:
        x, y = pose.to_world((local_mm[0] * MM, local_mm[1] * MM))
        return (float(x), float(y))

    tray = spec["tray"]
    centre = world_xy([0.5 * sum(tray["inner_x"]), 0.5 * sum(tray["inner_y"])])
    half = (0.5 * (tray["inner_x"][1] - tray["inner_x"][0]) * MM, 0.5 * (tray["inner_y"][1] - tray["inner_y"][0]) * MM)
    seats: dict[str, Seat] = {
        "tray": Seat(
            "tray", "tray", xy=centre,
            floor_z=oz + tray["floor_z"] * MM, top_z=oz + tray["wall_top_z"] * MM,
            x_range=(centre[0] - half[0], centre[0] + half[0]),
            y_range=(centre[1] - half[1], centre[1] + half[1]), pose=pose, half=half,
        ),
    }
    for bore in spec["bores"]:
        seats[bore["id"]] = Seat(
            bore["id"], "hole_vertical", world_xy(bore["xy"]), oz + bore["floor_z"] * MM, top,
            diameter=bore["diameter"] * MM, chamfer=bore["mouth_chamfer"] * MM, intended_part=bore["part"],
            pose=pose)
    for pocket in spec["hex_pockets"]:
        if pocket["flats_facing"] not in ("±y", "±y"):
            raise ValueError(f"unsupported hex pocket orientation {pocket['flats_facing']!r}")
        seats[pocket["id"]] = Seat(
            pocket["id"], "hex_pocket", world_xy(pocket["xy"]), oz + pocket["floor_z"] * MM, top,
            across_flats=pocket["across_flats"] * MM, chamfer=pocket["mouth_chamfer"] * MM,
            flat_normal_yaw=math.pi / 2 + pose.yaw, intended_part=pocket["part"], pose=pose)
    return seats


SEATS: dict[str, Seat] = seats_for(FIXTURE_SPEC)
BLOCK_TOP_Z = FIXTURE_ORIGIN[2] + FIXTURE_SPEC["block"]["top_z"] * MM
# World bounds of the nominal footprint.
FOOTPRINT_X = tuple(float(f(NOMINAL_POSE.corners()[:, 0])) for f in (np.min, np.max))
FOOTPRINT_Y = tuple(float(f(NOMINAL_POSE.corners()[:, 1])) for f in (np.min, np.max))
# The block face a part can lean on: the one facing world -y, the parts' side, at the nominal yaw
# (quarter turns of its outward normal from fixture +x; the -y face at yaw 0, the +x face at -90 deg).
LEAN_FACE = round((-math.pi / 2 - NOMINAL_YAW) / (math.pi / 2)) % 4


def lean_face(spec: dict, pose: FixturePose, u: float = 0.0) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """World xy of the lean face's point `u` along it, the horizontal normal into the face, and the
    face's extent in `u` (fixture metres)."""
    out = np.array([(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][LEAN_FACE])
    (x0, x1), (y0, y1) = ((v * MM for v in spec["block"][k]) for k in ("x", "y"))
    depth = (x1, y1, -x0, -y0)[LEAN_FACE]
    span = (y0, y1) if LEAN_FACE % 2 == 0 else (x0, x1)
    local = depth * out + u * np.abs(out[::-1])
    inward = -pose.to_world(out) + pose.to_world((0.0, 0.0))
    return pose.to_world(local), inward, span


# --------------------------------------------------------------------- geometry
# Everything below works in fixture-local millimetres, then converts once.

def _clip(poly: list[tuple[float, float]], nx: float, ny: float, c: float) -> list[tuple[float, float]]:
    """Keep the part of a convex polygon where nx*x + ny*y >= c (Sutherland-Hodgman)."""
    out: list[tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = nx * p[0] + ny * p[1] - c
        dq = nx * q[0] + ny * q[1] - c
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area(poly: list[tuple[float, float]]) -> float:
    return 0.5 * abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1])))


def _profile_pieces(half_x: float, half_y: float, normals: list[float], apothems: list[float]
                    ) -> list[list[list[tuple[float, float]]]]:
    """Square (half_x, half_y) minus a convex polygon, as one convex piece per polygon edge.

    The polygon is given by its edge normals (yaw, radians) and one apothem list
    per level (e.g. bottom and top of a chamfer). A piece is the wedge between
    the rays through that edge's two vertices, inside the square, outside the
    edge. Returns, per edge, the piece outline at every level.
    """
    k = len(normals)
    pieces: list[list[list[tuple[float, float]]]] = []
    square = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    for i in range(k):
        yaw = normals[i]
        lo = 0.5 * (normals[i - 1] + yaw) if i > 0 else yaw - 0.5 * (normals[1] - normals[0])
        hi = 0.5 * (yaw + normals[(i + 1) % k]) if i < k - 1 else yaw + 0.5 * (normals[1] - normals[0])
        levels = []
        for apothem in apothems:
            poly = _clip(square, -math.sin(lo), math.cos(lo), 0.0)  # left of the low vertex ray
            poly = _clip(poly, math.sin(hi), -math.cos(hi), 0.0)  # right of the high vertex ray
            poly = _clip(poly, math.cos(yaw), math.sin(yaw), apothem)  # outside the edge
            levels.append(poly)
        pieces.append(levels)
    return pieces


def _subtract_rects(bounds: tuple[float, float, float, float],
                    holes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Rectangles covering `bounds` minus `holes`. Each rect is (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = bounds
    xs = sorted({x0, x1, *[min(max(h[0], x0), x1) for h in holes], *[min(max(h[1], x0), x1) for h in holes]})
    rects: list[tuple[float, float, float, float]] = []
    for xa, xb in zip(xs, xs[1:]):
        if xb - xa < 1e-6:
            continue
        mid = 0.5 * (xa + xb)
        covered = sorted((h[2], h[3]) for h in holes if h[0] < mid < h[1])
        y = y0
        for cy0, cy1 in covered:
            if cy0 > y + 1e-6:
                rects.append((xa, xb, y, min(cy0, y1)))
            y = max(y, cy1)
        if y < y1 - 1e-6:
            rects.append((xa, xb, y, y1))
    return rects


@dataclass(frozen=True)
class _Feature:
    seat_id: str
    cx: float
    cy: float
    half_x: float
    half_y: float
    floor: float


def _features(fixture_spec: dict) -> list[_Feature]:
    out = []
    for bore in fixture_spec["bores"]:
        h = 0.5 * bore["diameter"] + bore["mouth_chamfer"] + _FEATURE_MARGIN_MM
        out.append(_Feature(bore["id"], *bore["xy"], h, h, bore["floor_z"]))
    for pocket in fixture_spec["hex_pockets"]:
        a = 0.5 * pocket["across_flats"] + pocket["mouth_chamfer"]
        corner = a / math.cos(math.pi / 6)
        out.append(_Feature(pocket["id"], *pocket["xy"], corner + _FEATURE_MARGIN_MM, a + _FEATURE_MARGIN_MM,
                            pocket["floor_z"]))
    return out


def collider_layout(fixture_spec: dict = FIXTURE_SPEC) -> dict[str, list]:
    """Fixture colliders in fixture-local mm: {'boxes': [(lo, hi)], 'prisms': [(bottom, z0, top, z1, cx, cy)]}.

    A box is (x0, y0, z0), (x1, y1, z1). A prism is a convex piece: its bottom
    outline at z0 and its top outline at z1, relative to the feature centre.
    """
    spec = fixture_spec
    fx, fy = spec["footprint"]
    X0, X1, Y0, Y1 = -fx / 2, fx / 2, -fy / 2, fy / 2
    floor = spec["base_floor_z"]
    tray = spec["tray"]
    wall, wall_top = tray["wall"], tray["wall_top_z"]
    bx0, bx1 = spec["block"]["x"]
    by0, by1 = spec["block"]["y"]
    top = spec["block"]["top_z"]
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
        ((X0, Y0, 0.0), (X1, Y1, floor)),  # base slab / tray floor
        ((X0, Y0, floor), (X0 + wall, Y1, wall_top)),  # tray front wall
        ((X0 + wall, Y0, floor), (bx0, Y0 + wall, wall_top)),  # tray side walls
        ((X0 + wall, Y1 - wall, floor), (bx0, Y1, wall_top)),
    ]
    feats = _features(fixture_spec)
    levels = sorted({floor, top, *(f.floor for f in feats)})
    for z_lo, z_hi in zip(levels, levels[1:]):
        cuts = [(f.cx - f.half_x, f.cx + f.half_x, f.cy - f.half_y, f.cy + f.half_y)
                for f in feats if f.floor <= z_lo + 1e-9]
        for rx0, rx1, ry0, ry1 in _subtract_rects((bx0, bx1, by0, by1), cuts):
            boxes.append(((rx0, ry0, z_lo), (rx1, ry1, z_hi)))
    prisms: list[tuple] = []
    by_id = {f.seat_id: f for f in feats}
    for bore in spec["bores"]:
        f = by_id[bore["id"]]
        r, ch = 0.5 * bore["diameter"], bore["mouth_chamfer"]
        normals = [2 * math.pi * (i + 0.5) / HOLE_SIDES for i in range(HOLE_SIDES)]
        for bottom, mouth, rim in _profile_pieces(f.half_x, f.half_y, normals, [r, r, r + ch]):
            if _area(bottom) > 1e-3:
                prisms.append((bottom, f.floor, bottom, top - ch, f.cx, f.cy))
            if ch > 0 and _area(mouth) > 1e-3 and _area(rim) > 1e-3:
                prisms.append((mouth, top - ch, rim, top, f.cx, f.cy))
    for pocket in spec["hex_pockets"]:
        f = by_id[pocket["id"]]
        a, ch = 0.5 * pocket["across_flats"], pocket["mouth_chamfer"]
        # Flats facing ±y: corners on ±x, edge normals at 30° + k·60°. The lead-in widens the
        # apothem by `ch` over the top `ch` (45° on the flats), like the printed pocket.
        normals = [math.radians(30 + 60 * i) for i in range(6)]
        for bottom, mouth, rim in _profile_pieces(f.half_x, f.half_y, normals, [a, a, a + ch]):
            if _area(bottom) > 1e-3:
                prisms.append((bottom, f.floor, bottom, top - ch, f.cx, f.cy))
            if ch > 0 and _area(mouth) > 1e-3 and _area(rim) > 1e-3:
                prisms.append((mouth, top - ch, rim, top, f.cx, f.cy))
    return {"boxes": boxes, "prisms": prisms}


FIXTURE_BODY = "fixture"


def add_fixture(spec: mujoco.MjSpec, fixture_spec: dict = FIXTURE_SPEC, body_mesh: Path = BODY_MESH) -> None:
    """Printed body + inlay (visual) and the convex colliders, on one mocap body at FIXTURE_ORIGIN.

    A mocap body is static for the physics (contacts never move it and it does not collide with
    the table) but its pose is per episode: `data.mocap_pos` / `mocap_quat` (see `FixturePose`).
    """
    wb = spec.worldbody.add_body(name=FIXTURE_BODY, mocap=True, pos=list(FIXTURE_ORIGIN),
                                 quat=NOMINAL_POSE.quat().tolist())
    ox, oy, oz = 0.0, 0.0, 0.0
    for name, path, rgba, specular in (("fixture_body", body_mesh, BODY_RGBA, 0.18),
                                       ("fixture_text", TEXT_MESH, TEXT_RGBA, 0.10)):
        spec.add_mesh(name=name, file=str(path), scale=[MM, MM, MM])
        mat = spec.add_material(name=f"mat_{name}")
        mat.rgba = list(rgba)
        mat.specular = specular
        mat.shininess = 0.3
        wb.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                    material=f"mat_{name}", pos=[ox, oy, oz], contype=0, conaffinity=0, density=0)
    hit = dict(rgba=[0.0, 0.0, 0.0, 0.0], friction=list(FIXTURE_FRICTION), solref=[0.01, 1.0],
               priority=FIXTURE_PRIORITY, condim=FIXTURE_CONDIM, group=3, density=0)
    layout = collider_layout(fixture_spec)
    for i, (lo, hi) in enumerate(layout["boxes"]):
        centre = [ox + 0.5 * (lo[0] + hi[0]) * MM, oy + 0.5 * (lo[1] + hi[1]) * MM, oz + 0.5 * (lo[2] + hi[2]) * MM]
        size = [0.5 * (hi[k] - lo[k]) * MM for k in range(3)]
        wb.add_geom(name=f"fixture_box_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre, size=size, **hit)
    for i, (bottom, z0, top, z1, cx, cy) in enumerate(layout["prisms"]):
        verts = [(x * MM, y * MM, z0 * MM) for x, y in bottom] + [(x * MM, y * MM, z1 * MM) for x, y in top]
        mesh = spec.add_mesh(name=f"fixture_piece_{i}")
        mesh.uservert = [c for v in verts for c in v]
        wb.add_geom(name=f"fixture_piece_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"fixture_piece_{i}",
                    pos=[ox + cx * MM, oy + cy * MM, oz], **hit)


def is_fixture_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    return name.startswith("fixture_")


def hex_fit_yaw(part_across_flats: float, pocket_across_flats: float) -> float:
    """Largest yaw error (rad, about the 60° symmetry) at which a centred hex part still clears the pocket."""
    corner = part_across_flats / math.sqrt(3.0)
    apothem = 0.5 * pocket_across_flats
    # A part corner at angle φ off a pocket corner sits inside when corner <= apothem / cos(30° - φ).
    ratio = min(1.0, apothem / corner)
    return max(0.0, math.radians(30.0) - math.acos(ratio))


def _sanity() -> np.ndarray:  # pragma: no cover - debugging aid
    layout = collider_layout()
    return np.array([len(layout["boxes"]), len(layout["prisms"])])
