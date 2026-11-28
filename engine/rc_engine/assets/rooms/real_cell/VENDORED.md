# real_cell

The real-arm cell at the event venue ("Dawn" office). It is the Reality Check team's own capture from
2026-11-27. Nothing in this folder is third-party.

Sources:

- Two Scaniverse LiDAR scans of the room (OBJ, MTL and JPEG texture), taken on an iPhone.
- iPhone photos:
  - `6.png`, `7.png` and `8.png` show the fixture and the tabletop.
  - `9.png` is a 2576 x 599 panorama taken from beside the robot.
  - `10.png` to `14.png` are close-ups of the walls. They were used only to check the panorama by eye.
- The two side cameras' 1080p frames (`left.jpg`, `right.jpg`) with their intrinsics and the arm's joints
  at capture (`intrinsics.json`, `capture.json`). These are the Gemini 336 cameras at their calibrated
  poses.

| File | What |
|---|---|
| `scan.obj`, `scan.png`, `scan2.obj`, `scan2.png` | The two scans, cut round the cell, decimated, textures at 2048 px. The floor round the cell is snapped to the sim's floor and removed under the table and the pedestal. |
| `proj_left.*`, `proj_right.*` | Room surfaces each side camera sees, textured from its frame. The robot, the tripods and a band round the table are inpainted, so nothing behind the arm shows grey. |
| `pano.obj`, `pano.png` | The room's bounding shell, textured with the panorama. It shows through holes in the scans. |
| `shell_floor.obj` | Shell floor below the panorama's view, in the scan's floor colour. |
| `room.json` | Materials, lighting, the panorama fit, projection stats and file hashes. |

Total size is about 23 MB and 107k triangles. The scans and photos themselves are not redistributed.

To reproduce (needs uv), from `engine/`:

```
MUJOCO_GL=cgl uv run --script scripts/convert_real_cell.py <scan folder> <scan 2 folder> <photo folder> \
    rc_engine/assets/rooms/real_cell --frames <frames folder> --panorama <photo folder>/9.png
```

The converter also rewrites `../../cell/cell.json` and keeps its calibration. To write the cell files elsewhere,
pass `--cell`. It runs `scripts/project_real_cell.py` last.

`PANO_FIT` in that script was fitted offline. Its centre, yaw, angular period, vertical scale and horizon
were chosen to maximise the correlation between the panorama and two things: the side frames' room pixels
(placed in 3D by the room's depth) and cylindrical renders of the scan. The fit is good to about
10 to 20 degrees.
