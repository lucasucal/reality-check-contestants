"""Reality Check eval fixture v3: real geometry (mm) -> fixture_body.stl, fixture_text.stl, fixture_v3.json.

Frame: origin = footprint centre at table level, +x away from the robot, z up.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trophy"))
import text2d as T  # noqa: E402

# ---------------------------------------------------------------- parameters (mm)
FOOT_X, FOOT_Y = 250.0, 230.0
X0, X1, Y0, Y1 = -FOOT_X / 2, FOOT_X / 2, -FOOT_Y / 2, FOOT_Y / 2
FLOOR = 4.0  # tray floor / base slab
WALL = 4.0
WALL_H = 22.0
TRAY_INNER_X = 95.0
BLOCK_X0 = X0 + WALL + TRAY_INNER_X  # block front face = tray back wall
TOP = 75.0
EDGE_CH = 2.0
WEB = 10.0
INLAY = 0.8

HOLE_CH = 5.0
HEX_CH = 2.0
# (id, bore, depth, part) — y-order top (+y) to bottom (-y): not monotonic
HOLES = [("bore_m24", 28.0, 55.0, "M24x50"), ("bore_m16", 20.0, 55.0, "M16x50"), ("bore_m30", 35.0, 65.0, "M30x60")]
# (id, across-flats, depth, part) — ISO 4032: M24 s36 m21.5, M30 s46 m25.6, M36 s55 m31
HEXES = [("hex_m30", 51.0, 13.0, "M30 nut"), ("hex_m36", 60.0, 15.0, "M36 nut"), ("hex_m24", 41.0, 10.0, "M24 nut")]


def column_layout(extents_y, y_lo, y_hi):
    """Centres for features of given y-extents spread evenly (equal gaps) between y_hi and y_lo."""
    gap = (y_hi - y_lo - sum(extents_y)) / (len(extents_y) - 1)
    ys, y = [], y_hi
    for e in extents_y:
        ys.append(y - e / 2)
        y -= e + gap
    return ys, gap


def hex_poly(af: float, cx=0.0, cy=0.0) -> Polygon:
    """Hexagon with flats facing ±y (y-extent = af, x-extent = corners)."""
    r = af / math.sqrt(3)
    return Polygon([(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in range(0, 360, 60)])


def layout():
    hole_env = [d + 2 * HOLE_CH for _, d, _, _ in HOLES]
    hex_env_y = [af + 2 * HEX_CH for _, af, _, _ in HEXES]
    hex_env_x = [(af + 2 * HEX_CH) * 2 / math.sqrt(3) for _, af, _, _ in HEXES]
    ylo, yhi = Y0 + WEB, Y1 - WEB
    hy, hgap = column_layout(hole_env, ylo, yhi)
    xy_, xgap = column_layout(hex_env_y, ylo, yhi)
    # columns in x: block front edge, web, hole column, web, hex column, web, back edge; spread slack evenly
    wh, wx = max(hole_env), max(hex_env_x)
    slack = (X1 - BLOCK_X0) - (3 * WEB + wh + wx)
    assert slack >= 0, f"block too shallow by {-slack:.1f} mm"
    web_x = WEB + slack / 3
    hx = BLOCK_X0 + web_x + wh / 2
    xx = hx + wh / 2 + web_x + wx / 2
    holes = [dict(id=i, kind="bore", xy=[round(hx, 2), round(y, 2)], diameter=d, depth=dp, mouth_chamfer=HOLE_CH,
                  floor_z=TOP - dp, part=p) for (i, d, dp, p), y in zip(HOLES, hy)]
    hexes = [dict(id=i, kind="hex_pocket", xy=[round(xx, 2), round(y, 2)], across_flats=af, depth=dp,
                  mouth_chamfer=HEX_CH, floor_z=TOP - dp, flats_facing="±y", part=p) for (i, af, dp, p), y in zip(HEXES, xy_)]
    return holes, hexes


# ---------------------------------------------------------------- geometry


def box3(x0, y0, z0, x1, y1, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cs(geom):
    polys = getattr(geom, "geoms", [geom])
    rings = []
    for p in polys:
        if p.is_empty:
            continue
        rings.append(np.asarray(p.exterior.coords)[:-1])
        rings += [np.asarray(r.coords)[:-1] for r in p.interiors]
    return mf.CrossSection(rings, mf.FillRule.EvenOdd)


def tri(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)


def build():
    holes, hexes = layout()
    # Base slab + tray walls (-x, ±y) + raised block with chamfered top edges
    body = box3(X0, Y0, 0, X1, Y1, FLOOR)
    body += box3(X0, Y0, 0, X0 + WALL, Y1, WALL_H)
    body += box3(X0, Y0, 0, BLOCK_X0, Y0 + WALL, WALL_H)
    body += box3(X0, Y1 - WALL, 0, BLOCK_X0, Y1, WALL_H)
    block = mf.Manifold.batch_hull([box3(BLOCK_X0, Y0, 0, X1, Y1, TOP - EDGE_CH),
                                    box3(BLOCK_X0 + EDGE_CH, Y0 + EDGE_CH, TOP - EDGE_CH, X1 - EDGE_CH, Y1 - EDGE_CH, TOP)])
    body += block
    cuts = []
    for h in holes:
        x, y = h["xy"]
        r = h["diameter"] / 2
        cuts.append(mf.Manifold.cylinder(h["depth"] + 1, r, r, 96).translate([x, y, h["floor_z"]]))
        cuts.append(mf.Manifold.cylinder(HOLE_CH + 1e-3, r, r + HOLE_CH, 96).translate([x, y, TOP - HOLE_CH]))
        cuts.append(mf.Manifold.cylinder(1, r + HOLE_CH, r + HOLE_CH, 96).translate([x, y, TOP]))
    for p in hexes:
        x, y = p["xy"]
        af = p["across_flats"]
        hx = cs(hex_poly(af))
        cuts.append(mf.Manifold.extrude(hx, p["depth"] + 1).translate([x, y, p["floor_z"]]))
        k = (af + 2 * HEX_CH) / af
        cuts.append(mf.Manifold.extrude(hx, HEX_CH + 1e-3, scale_top=[k, k]).translate([x, y, TOP - HEX_CH]))
        cuts.append(mf.Manifold.extrude(cs(hex_poly(af + 2 * HEX_CH)), 1).translate([x, y, TOP]))
    body = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)

    # REALITY / CHECK on both block side faces (±y), Archivo wide black, inlaid 0.8 mm
    A = T.font("Archivo", wdth=125, wght=900)
    face_w = X1 - BLOCK_X0
    cap = 15.0
    l1, l2 = T.text("REALITY", cap, A), T.text("CHECK", cap, A)
    while max(T.width(l1), T.width(l2)) > face_w - 2 * 14:
        cap -= 0.5
        l1, l2 = T.text("REALITY", cap, A), T.text("CHECK", cap, A)
    lead = cap * 0.45
    zc = (FLOOR + TOP) / 2 + 2
    g = T.place(l1, 0, lead / 2, "bc").union(T.place(l2, 0, -lead / 2, "tc"))
    xc = (BLOCK_X0 + X1) / 2
    slab = mf.Manifold.extrude(cs(g), INLAY + 0.01)  # u, v, w in [0, INLAY]
    # +y face (viewer at +y): u->-x, v->z, w->+y ; -y face: u->+x, v->z, w->-y (both det +1)
    plus = slab.transform([[-1, 0, 0, xc], [0, 0, 1, Y1 - INLAY], [0, 1, 0, zc]])
    minus = slab.transform([[1, 0, 0, xc], [0, 0, -1, Y0 + INLAY], [0, 1, 0, zc]])
    text = (plus + minus) ^ body
    body = body - text
    return body, text, holes, hexes, dict(cap=cap)


def checks(holes, hexes):
    """Min web between feature mouths (top-face outlines) and to block edges."""
    outl = {}
    for h in holes:
        outl[h["id"]] = Point(*h["xy"]).buffer(h["diameter"] / 2 + HOLE_CH, 128)
    for p in hexes:
        outl[p["id"]] = hex_poly(p["across_flats"] + 2 * HEX_CH, *p["xy"])
    flat_top = box(BLOCK_X0 + EDGE_CH, Y0 + EDGE_CH, X1 - EDGE_CH, Y1 - EDGE_CH)
    ids = list(outl)
    pair = min(((outl[a].distance(outl[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]))
    edge = min((flat_top.exterior.distance(outl[a]), a) for a in ids)
    edge_outer = min((box(BLOCK_X0, Y0, X1, Y1).exterior.distance(outl[a]), a) for a in ids)
    return pair, edge, edge_outer


if __name__ == "__main__":
    body, text, holes, hexes, info = build()
    tb, tt = tri(body), tri(text)
    tb.export(HERE / "fixture_body.stl")
    tt.export(HERE / "fixture_text.stl")
    pair, edge, edge_outer = checks(holes, hexes)
    b = tb.bounds
    print("bounds", np.round(b, 2).tolist(), "bed margin x", (256 - (b[1][0] - b[0][0])) / 2, "y", (256 - (b[1][1] - b[0][1])) / 2)
    print("body watertight", tb.is_watertight, tb.is_volume, "faces", len(tb.faces), "vol cm3", round(tb.volume / 1000, 1))
    print("text watertight", tt.is_watertight, tt.is_volume, "vol cm3", round(tt.volume / 1000, 2), "cap", info["cap"])
    print("overlap body^text mm3", round((body ^ text).volume(), 3))
    print("min web feature-feature", round(pair[0], 2), pair[1:], "| to chamfer edge", round(edge[0], 2), edge[1],
          "| to outer edge", round(edge_outer[0], 2))
    for f in holes + hexes:
        print(f)
    spec = {
        "_note": "Fixture v3 PROPOSAL. Millimetres, fixture-local: origin = footprint centre at table level, +x away from robot, z up.",
        "footprint": [FOOT_X, FOOT_Y], "height": TOP, "base_floor_z": FLOOR,
        "tray": {"inner_x": [X0 + WALL, BLOCK_X0], "inner_y": [Y0 + WALL, Y1 - WALL], "wall": WALL,
                 "wall_top_z": WALL_H, "floor_z": FLOOR, "back_wall": "block front face"},
        "block": {"x": [BLOCK_X0, X1], "y": [Y0, Y1], "top_z": TOP, "edge_chamfer": EDGE_CH},
        "bores": holes, "hex_pockets": hexes,
        "lettering": {"text": ["REALITY", "CHECK"], "font": "Archivo wdth125 wght900", "cap_height": info["cap"],
                      "faces": ["+y", "-y"], "inlay_depth": INLAY, "body": "fixture_text.stl"},
        "print": {"orientation": "as modelled, base on bed", "supports": "none", "bed": "Bambu X2D 256x256"},
    }
    (HERE / "fixture_v3.json").write_text(json.dumps(spec, indent=2))
