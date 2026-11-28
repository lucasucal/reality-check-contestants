# Fastener kit: SimReady Hardware (CC-BY-4.0)

The bolts, nuts and washers are real ISO/DIN hardware from the **SimReady Hardware** dataset,
converted for MuJoCo.

- Source: <https://huggingface.co/datasets/projectsim/simready-hardware> (dataset)
- Revision: `827f45cdffe63ae699351fdba58c50de73e4d40a`
- Licence: CC-BY-4.0
- Converter: `engine/scripts/convert_simready.py` (this directory is its output; nothing here is
  edited by hand)

**Attribution.** "SimReady Hardware" by projectsim, <https://huggingface.co/datasets/projectsim/simready-hardware>,
licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>). Converted to OBJ, decimated
and given simulator colliders by the Reality Check team (see below). Part dimensions in the dataset
follow the ISO/DIN standards as tabulated in the FreeCAD Fasteners Workbench.

## Reproduce

```sh
cd engine
# Download only the packages the kit uses into an empty directory of its own (about 11 MB). Treat it as
# data: never run anything from it.
uv run --group simready python scripts/convert_simready.py download DL_DIR
uv run --group simready python -I scripts/convert_simready.py convert DL_DIR          # rewrite this directory
uv run --group simready python -I scripts/convert_simready.py convert DL_DIR --check  # compare, write nothing
```

`--check` passes on the vendored files (geometry within 1 µm, manifest identical).

## Parts

Dimensions in mm (from each package's sidecar JSON). Every part is grey PLA at an estimated 680 kg/m³
(20% infill, 3 walls; `tabletop.PART_DENSITY`, `PART_LOOK`), like the printed event kit; the mass in g is
the dataset volume x that density. `manifest.json` keeps the dataset's own materials and masses.

| Kit id | Standard | Dataset package stem | Dimensions | Visual triangles (source -> vendored) | Mass (g) |
|---|---|---|---|---|---|
| `bolt_m16x50` | ISO 4017 M16 | `bolt_iso4017_m16x2_l50` | d 16, L 50, k 10, s 24 | 17142 -> 5998 | 9.2 |
| `bolt_m24x70` | ISO 4017 M24 | `bolt_iso4017_m24x3_l70` | d 24, L 70, k 15, s 36 | 17684 -> 6000 | 29.7 |
| `bolt_m30x100` | ISO 4017 M30 | `bolt_iso4017_m30x3p5_l100` | d 30, L 100, k 18.7, s 46 | 24524 -> 6000 | 64.5 |
| `nut_m24` | ISO 4032 M24 | `nut_iso4032_m24x3` | s 36, m 21.5 | 11054 -> 4000 | 10.6 |
| `nut_m30` | ISO 4032 M30 | `nut_iso4032_m30x3p5` | s 46, m 25.6 | 11744 -> 4000 | 21.0 |
| `nut_m36` | ISO 4032 M36 | `nut_iso4032_m36x4` | s 55, m 31 | 12720 -> 3998 | 36.2 |
| `washer_m33` | DIN 125-A M33 | `washer_din125a_m33` | ID 34, OD 60, t 5 | 1008 | 6.5 |
| `nut_m12` | ISO 4032 M12 | `nut_iso4032_m12x1p75` | s 18, m 10.8 | 7994 -> 2500 | 1.4 |
| `nut_m16` | ISO 4032 M16 | `nut_iso4032_m16x2` | s 24, m 14.8 | 11128 -> 2498 | 3.3 |
| `nut_m20` | ISO 4032 M20 | `nut_iso4032_m20x2p5` | s 30, m 18 | 11272 -> 2500 | 6.2 |
| `washer_m12` | ISO 7089 M12 | `washer_iso7089_m12` | ID 13, OD 24, t 2.5 | 1008 | 0.5 |
| `washer_m16` | ISO 7089 M16 | `washer_iso7089_m16` | ID 17, OD 30, t 3 | 1008 | 1.0 |
| `washer_m20` | ISO 7089 M20 | `washer_iso7089_m20` | ID 21, OD 37, t 3 | 1008 | 1.5 |

The first seven are task parts (the fixture's three holes and three hex pockets, and the washer for the
35 mm hole); the M12-M20 nuts and washers are clutter. 19 OBJ files (13 visual meshes, 6 nut hulls), 1.3 MB.

## Friction

The dataset's contact friction is not used: the simulator keeps its own part, pad, table and
printed-fixture friction.

## What the conversion does

- Reads each package's `visual` mesh (metres, Z up, checked), triangulates it and moves it into the
  simulator's body frame: a bolt along +X with its head at +X and the origin at the middle of its full
  length (head + shank); a nut or washer Z up about its centre, a nut's flats facing its body ±x.
- Checks the source mesh's bounds against the sidecar dimensions (0.5 mm), decimates the visual mesh to
  the part's triangle budget (quadric, `fast_simplification`; the cosmetic threads stay in the visual
  mesh only), clamps it to the collider box and checks the bounds again (worst 0.41 mm, `nut_m20`).
- Colliders (`manifest.json` `colliders`; the simulator builds them in `tabletop._add_part`), every
  dimension from the sidecar: a bolt is a hexagonal prism head (s across flats, k high) and a shank
  cylinder (d, L); a washer is a ring of 16 boxes (ID, OD, t); a nut is the convex hull in
  `<id>_hull.obj`: its two faces are the bearing circle measured on the mesh (a 24-gon) and its six
  corners stand at s/√3 as far up as the mesh's corners reach (the ISO 30° corner chamfer), 60 vertices,
  within 0.12% of the volume of the full mesh's own hull. The raw hull's hundreds of near-coplanar facets
  made a small dense nut (a stainless M12) chatter up to 4 mm on the table; the hull of the decimated
  mesh rested a nut up to 1 mm low.
- Mass properties from the sidecars: the volume, the centre of mass
  and the principal inertia, turned into the body frame and stored per unit density
  (`inertia_per_density`), so mass = volume x density and inertia = that x density, with the
  simulator's one density (`tabletop.PART_DENSITY`).
