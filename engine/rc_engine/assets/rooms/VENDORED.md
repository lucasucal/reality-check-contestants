# Rooms

Converted from [SimReady Hardware](https://huggingface.co/datasets/projectsim/simready-hardware)
(dataset revision `827f45cdffe63ae699351fdba58c50de73e4d40a`), licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: "SimReady Hardware" by
projectsim, CC-BY-4.0. Changes: flattened from USD, placed around the Reality Check bench, merged per
material, decimated, textures reduced to 512 px PNG, normal/roughness/metallic maps dropped, PBR
values approximated by MuJoCo materials, the dome HDRI baked to a cube map. The originals are not
redistributed here.

| Folder | Source | Triangles (source -> here) | Size |
|---|---|---|---|
| `workshop/` | `scenes/workshop/workshop.usda` (room, workbench, stool) | 243,448 -> 49,996 | 6.7 MB |
| `office/` | `scenes/carpeted_office_furnished/CarpetedOfficeFurnished.usdc` (room, desk, chair, vase) | 155,476 -> 49,998 | 6.6 MB |
| `industrial_bay/` | `scenes/industrial_bay_walkaround/IndustrialBayWalkaround.usdc` (bay, two yellow benches) | 90,043 -> 49,999 | 5.6 MB |
| `exterior_sky.png` | the scenes' dome HDRI, `tears_of_steel_bridge_4k.hdr` (the same file in all three), 1024 px faces | - | 7.3 MB |

Each room folder holds one OBJ per material (engine frame: robot base axis at the origin, table top z = 0,
room floor z = -0.75; the engine lowers them to the real cell's floor), its base-colour PNGs and `room.json`
(placement, materials, the scene's lights summarised, file hashes). The converter also writes a
toolbox prop (`props/toolbox_yellow`), which is not shipped. How the engine uses them:
`rc_engine/tasks/rooms.py`.

To reproduce (byte-identical; needs uv):

```
python -I -c 'from huggingface_hub import snapshot_download; snapshot_download("projectsim/simready-hardware", repo_type="dataset", revision="827f45cdffe63ae699351fdba58c50de73e4d40a", allow_patterns=["scenes/**", "props/toolbox_yellow/**", "README.md"], local_dir="/tmp/simready")'
uv run --script engine/scripts/convert_rooms.py /tmp/simready engine/rc_engine/assets/rooms
```

The converter pins usd-core 26.11, pymeshlab 2025.7.post1, NumPy 2.4.6 and Pillow 13.0.0 (PEP 723
header). Placements, budgets and the light mapping are constants at the top of the script.

## `real_cell/`: the venue around the real-arm cell

The Reality Check team's own capture of the venue (no third-party licence): see
[`real_cell/VENDORED.md`](real_cell/VENDORED.md). The cell's own geometry (table, pedestal, fixture,
cameras) is `../../cell/cell.json`, measured from the same scans.
