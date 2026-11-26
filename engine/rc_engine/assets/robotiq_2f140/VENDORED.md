# 2F-140 model provenance

The MJCF `2f140.xml` and `assets/pad.stl` come from
[personalrobotics/geodude_assets](https://github.com/personalrobotics/geodude_assets)
commit `f130a8f435ef7780fbc28014805c25cce6acd67b`,
`src/geodude_assets/models/robotiq_2f140`. The third-party MJCF is derived from
the ROS-Industrial 2F-140 URDF and remains the source of the articulation,
inertias, force limits, contacts, controller, and pad. Its BSD-2-Clause license
is in `LICENSE`. The engine TCP `gripperframe` copies the model's `grasp_site`
(palm, TSR yaw). Neither that frame nor the model is calibrated to our cell.

The other six STL files are the exact visual mesh bytes from the Robotiq-maintained
[robotiq/robotiq-ros](https://github.com/robotiq/robotiq-ros) commit
`31c57bbb052063d6b6c3e9602dbc534db1e9831b`, package
`grippers/robotiq_description/meshes/visual/2f_140`. The package declares
BSD-3-Clause; the repository root copyright notice is retained as
`LICENSE-ROBOTIQ-ROS` and the grippers subtree's PickNik Robotics notice as
`LICENSE-PICKNIK`. Robotiq describes this package as a continuation of the
PickNik ROS 2 package. These files are distributed by Robotiq; that alone does
not establish that their geometry was created by the manufacturer or measured
on our gripper.

| Local file | Upstream suffix `robotiq_2f_140_…` | SHA-256 |
|---|---|---|
| `base.stl` | `base_link.stl` | `74a62de75ae10cf77c60f2c49749b5d11f4c265f8624bbe7697a941fa86f6b3b` |
| `base_mount.stl` | `coupling.stl` | `4281e83002a25c20dc07c68b8d77da30a13e9a8401f157f6848ed8287d7cce44` |
| `driver.stl` | `outer_knuckle.stl` | `f042edc44ede9773e7cad00f7d7354d0b7c1c6a6353fe897b2ca2e6ac71107fc` |
| `coupler.stl` | `outer_finger.stl` | `666a92ee075f6f320ffb13b39995a5b374657cee90ded4c52c23ede20a812f34` |
| `spring_link.stl` | `inner_knuckle.stl` | `d61e9c5304d8015333856e4a26d99e32b695b6ada993253986b1afe8e396ab11` |
| `follower.stl` | `inner_finger.stl` | `be21c5e18c901d4747cbc8fa1a2af15dad7173ff701482a8517e33278432bb21` |

Comparing binary STL triangles before replacement, all vertex coordinates in
five pairs were exactly equal and in the coupling pair differed by at most
`5.75e-23` in the unscaled millimetre file. Their raw file hashes differed.
The existing MJCF scales the coupling by 0.001 and places meshes on the same
link frames; no kinematic or collision model was changed. As before, MuJoCo
uses these visual meshes as the model's mesh collision geometry and the
independent pad boxes for gripping. The pinned ROS package also supplies
separate, lower-triangle-count collision STLs; these are different and were not
substituted or validated in the simulated linkage. The physical mount offset
is still an unmeasured, configurable assumption; the ROS coupling mesh is not
proof of the eventual installed coupling.

## Local modification

`2f140.xml` differs from the pinned upstream file in the drive only: the
actuator's velocity gain is removed (`biasprm` `0 -1000 -300` → `0 -1000 0`)
and the split tendon gets `damping="4"` instead. Inside the force-clamped
actuator the velocity term made the drive chatter at the 5 ms physics step in
contact (pad force pulsing about 1–28 N); as passive tendon damping it is
integrated implicitly and the grip is a steady ~46 N per pad at the same 6.2
cap and about the same finger speed. The comment above the actuator and
`docs/simulation-fidelity.md` (Gripper) give the measurements;
`engine/tests/test_embodiment.py::test_gripper_drive_is_steady_in_contact`
guards it.
