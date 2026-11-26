Vendored from [Universal Robots ROS 2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/0eb1a861f7772fceb38510d90fa2f98ec668070c)
(commit `0eb1a861f7772fceb38510d90fa2f98ec668070c`, package `ur_description` 4.4.1).
UR12e kinematics, source joint limits and inertias are from `config/ur12e/*`. The executable MJCF overrides the source elbow range with ±160° from the [current UR12e specifications](https://www.universal-robots.com/manuals/EN/HTML/SW10_12_1/Content/prod-usr-man/complianceUR12e/H_g5_sections/appendix_g5/tech_spec_data_12e.htm); the [pinned source](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/0eb1a861f7772fceb38510d90fa2f98ec668070c/config/ur12e/joint_limits.yaml) describes its ±180° bound as an artificial planning limit. Its
[`visual_parameters.yaml`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/0eb1a861f7772fceb38510d90fa2f98ec668070c/config/ur12e/visual_parameters.yaml)
selects `meshes/ur10e/visual/{base,shoulder,upperarm,forearm,wrist1,wrist2,wrist3}.dae`
and the corresponding collision STLs. The source's mesh offsets are retained in
`ur12e.xml`. The upstream [README](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/0eb1a861f7772fceb38510d90fa2f98ec668070c/README.md#license)
licenses these UR10e meshes under BSD-3-Clause (`LICENSE`); its separate Graphical
Documentation terms apply to the listed newer models, not UR10e.

MuJoCo does not load Collada. `convert_visuals.py` converts the seven source DAE
files to 20 per-material OBJ files (129,621 triangles, approximately 12 MiB)
using Python's standard library:

```sh
python3 engine/rc_engine/assets/ur12e/convert_visuals.py /path/to/Universal_Robots_ROS2_Description/meshes/ur10e/visual
```

It applies the source scene matrices to positions and normals, preserves triangle
indices and diffuse colors, groups by bound material, and ignores line primitives.
The pinned scene matrices have uniform scale. OBJ coordinates are in metres; no
simplification is performed. Diffuse RGBA values are specified on the visual geoms;
other Collada shading properties are not reproduced. Bounding boxes of converted
meshes match the official collision STLs to within 0.6 mm per axis. This is a
conversion check, not a measurement of the installed robot or its calibration.

Source DAE SHA-256 checksums:

```text
base      28226d95af8c1579a49fe69b62a00d9133a2a05258dad550bb9b583c80dc938f
shoulder  1c38a6afedfb8f064117da87b60c2dcb9ccd107f957e4c5e0a198f4c2bb63d36
upperarm  f9d2380d14127440f4a50ed79ab4e7562f86289cd6131016c3d49a3e7483e6ed
forearm   24d909834c380cceb0490181dae40bcdfcfa4a7fb0c6807d6e58794cdfadcf6e
wrist1    643717d042920313dd9daa81511bd9afa66a877625967e6dd8d359ec851b4237
wrist2    b820e8c8a4488563c7a21fe73bf2fa4038ec9f882dcf9db12e2d1df928253793
wrist3    52ace7588f131dc43b49ba88d1c83c1f871e6d00659ea84b1f437e2a7f1ab7fe
```

The official xacro (`urdf/ur_macro.xacro` + YAML) was flattened to `ur12e.urdf`
and compiled with `mujoco.MjSpec.from_file` to `ur12e.xml`. Collision STL geoms
are in hidden group 3; the new group 1 meshes have zero density and contact
masks. `rc_engine/embodiment.py` attaches the 2F-140, disables contacts on the
arm meshes and adds hand-fit capsule collisions, sets arm `gravcomp`, and retunes
the position PD. Effort limits are the UR12e datasheet values. The legacy `cup_*`
tasks select the original collision-STL visual shells from the same MJCF,
preserving camera shadows for the existing RGB-segmentation demo. The tabletop
and bench scenes use the converted visuals. The 2F-140 is vendored separately.
