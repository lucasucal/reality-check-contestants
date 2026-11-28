"""The live cell. RC_EMBODIMENT selects a registered id; only ur12e_robotiq_2f140 exists today.

The UR12e URDF has no motors. robot_spec() adds the cabinet analogue a host computer
would see when streaming servoj: gravity compensation on the arm (UR set_payload /
inverse-dynamics hold), a split position PD with kv from this arm's mass matrix, and
capsule collisions so the official overlapping STLs stay visual-only. cups.py upsamples
20 Hz joint_delta to the 5 ms physics step with UR's default 0.1 s lookahead.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

from .cell import WRIST_FOVY
from .schema import NUM_JOINTS

DEFAULT = "ur12e_robotiq_2f140"
REGISTERED = frozenset({DEFAULT})

ASSETS = Path(__file__).resolve().parent / "assets"
UR12E_XML = ASSETS / "ur12e" / "ur12e.xml"
GRIPPER_XML = ASSETS / "robotiq_2f140" / "2f140.xml"

# 6 arm hinges, then the 2F-140 (attached). Gripper command is a separate actuator.
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_BODIES = (
    "base_link",
    "base_link_inertia",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "flange",
    "tool0",
)
ARM_MESH_BODIES = ARM_BODIES[1:8]  # inertial base through wrist_3; skip empty base_link / flange / tool0

FLANGE_BODY = "flange"
# ROS-Industrial tool frame: tool0 +z is the tool axis (flange +x). The 2F-140 bolts
# on coaxially, so its approach is tool0 +z and its fingers open along tool0 ±x.
TOOL_BODY = "tool0"
GRIPPER_PREFIX = "gripper/"
GRIPPER_ACTUATOR = "gripper/fingers_actuator"
GRIPPER_JOINT = "gripper/right_driver_joint"
GRIPPERFRAME = "gripperframe"
# Geodude grasp_site: palm, TSR (z = approach, y = finger-open). Matches the 2F-140 TCP
# a host IK stack uses; 0.148 m was an ad-hoc pad site with no yaw.
GRIPPERFRAME_POS = (0.0, 0.0, 0.100)
GRIPPERFRAME_QUAT = (0.7071, 0.0, 0.0, -0.7071)  # Rz(-90°)

# Team adapter bracket (assets/wrist_mount): UR flange face -> Robotiq coupling face,
# measured on the Onshape part. Bracket mm map to tool0 m as (p - [0, 0, 6.4]) / 1000.
WRIST_MOUNT_STL = ASSETS / "wrist_mount" / "Bracket_G336_M3.stl"
BRACKET_FLANGE_Z_MM = 6.4
GRIPPER_SPACER_M = 0.0244

# The wrist camera on the bracket arm, in tool0. It looks along (0, -0.38268, 0.92388): the
# approach tilted 22.5° in toward the fingers, with image-up away from the tool axis, so the
# fingers sit at the bottom of the frame. The installed unit is an Orbbec Gemini 335Lg (the
# housing below is still the 336's): 640x360 (16:9) stream, factory fy 305.2 px, so 61.1°
# vertical (cell.json; pinhole, no distortion).
WRIST_CAMERA = "wrist"
WRIST_CAMERA_POS = (0.0140, 0.1103, 0.0649)
WRIST_CAMERA_XYAXES = (-1.0, 0.0, 0.0, 0.0, 0.92388, 0.38268)
WRIST_CAMERA_FOVY = WRIST_FOVY
WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 360
# Gemini 336 housing (89.5 x 25 x 30.7 mm), back face on the bracket camera plate.
_PLATE_NORMAL = (0.0, -0.38268, 0.92388)
_PLATE_CENTRE = (0.00027, 0.12209, 0.04292 - BRACKET_FLANGE_Z_MM / 1000)
CAMERA_HOUSING_HALF = (0.04475, 0.0125, 0.01535)
CAMERA_HOUSING_POS = tuple(c + n * CAMERA_HOUSING_HALF[2] for c, n in zip(_PLATE_CENTRE, _PLATE_NORMAL))
CAMERA_HOUSING_QUAT = (0.9807853, 0.1950903, 0.0, 0.0)  # Rx(22.5°): axes x, (0, .924, .383), n

# Match the 2026-11-28 START_POSE in the hackathon policy runner.
HOME_Q = (-2.932028, -1.278959, -2.076159, -1.365526, 1.570474, 0.615173307)
assert len(HOME_Q) == NUM_JOINTS

# Polar sampling around the base, in the reachable patch under HOME_Q.
CUP_R = (0.75, 0.92)
TARGET_R = (0.75, 0.92)
TOP_CAMERA_POS = (0.85, 0.0, 1.25)
TOP_CAMERA_XYAXES = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0)  # image-up = +x, image-right = -y
TOP_CAMERA_FOVY = 55.0

# Cabinet analogue. kp matches Menagerie proximal/wrist split; kv = 2 ζ √(kp J_eff)
# with ζ = 0.7 and J_eff = max diag(M) over home / stretch / zero (armature 0.1 included).
# Official UR12e forcerange stays on the actuators in ur12e.xml.
ARM_KP = (2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0)
ARM_KV = (237.0, 232.0, 105.0, 13.0, 11.0, 10.0)
assert len(ARM_KP) == NUM_JOINTS and len(ARM_KV) == NUM_JOINTS

PHYSICS_DT = 0.005  # s; cups.py keeps option.timestep in sync
SERVOJ_LOOKAHEAD = 0.1  # s; UR servoj default. 20 Hz Δq is interpolated across substeps.


def resolve(name: str | None = None) -> str:
    """Return the registered embodiment id. Unknown ids fail; the env var is kept for later."""
    raw = (name if name is not None else os.environ.get("RC_EMBODIMENT", "")).strip() or DEFAULT
    if raw not in REGISTERED:
        raise ValueError(f"unknown RC_EMBODIMENT={raw!r}; registered: {', '.join(sorted(REGISTERED))}")
    os.environ["RC_EMBODIMENT"] = raw
    return raw


def robot_spec(*, legacy_cup_visuals: bool = False) -> mujoco.MjSpec:
    """UR12e + 2F-140 as an MjSpec. Does not load SO-101. Caller adds the table/cups overlay."""
    resolve()
    spec = mujoco.MjSpec.from_file(str(UR12E_XML))
    if legacy_cup_visuals:
        _use_legacy_cup_visuals(spec)
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    # Flange face to Robotiq coupling face along tool0 +z. The default is the team
    # bracket; the override stays for trial couplers. It moves only the gripper.
    spacer = float(os.environ.get("RC_GRIPPER_SPACER_M", str(GRIPPER_SPACER_M)))
    if not 0 <= spacer <= 0.2:
        raise ValueError("RC_GRIPPER_SPACER_M must be between 0 and 0.2")
    tool = spec.body(TOOL_BODY)
    if spacer > 0:
        # The bracket's coupling boss (Ø63 mm), flange face to coupling. No mass.
        tool.add_geom(
            name="spacer_collision", type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            pos=[0, 0, spacer / 2], size=[0.0315, spacer / 2, 0], **_MOUNT_COLLISION)
    # Identity attach frame: base_mount +z (approach) is tool0 +z, and the fingers
    # open along tool0 ±x, perpendicular to the camera arm on tool0 +y.
    frame = tool.add_frame(name="gripper_mount", pos=[0, 0, spacer])
    # The 2F-140's own options, set before attaching: MuJoCo 4 warns (and logs) when they differ.
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10
    spec.attach(gripper, prefix=GRIPPER_PREFIX, frame=frame)
    spec.body(f"{GRIPPER_PREFIX}base_mount").add_site(
        name=GRIPPERFRAME, pos=list(GRIPPERFRAME_POS), quat=list(GRIPPERFRAME_QUAT))
    _add_wrist_mount(spec)
    spec.option.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10
    spec.option.timestep = PHYSICS_DT
    _apply_cabinet(spec)
    _apply_arm_collisions(spec)
    return spec


# Same contact settings as the arm capsules in _apply_arm_collisions.
_MOUNT_COLLISION = dict(
    contype=1, conaffinity=1, condim=3, density=0, group=3, friction=[0.5, 0.005, 0.0001])


def _add_wrist_mount(spec: mujoco.MjSpec) -> None:
    """Team bracket and Gemini 336 on tool0: visual mesh and box, collision primitives, camera.

    Everything here is welded to wrist_3, so MuJoCo already skips wrist_3, wrist_2
    and the gripper palm. The explicit excludes keep the mount from touching the
    rest of the arm and the fingers; it still collides with the table and parts.
    """
    tool = spec.body(TOOL_BODY)
    spec.add_mesh(name="wrist_mount", file=str(WRIST_MOUNT_STL), scale=[0.001, 0.001, 0.001])
    tool.add_geom(
        name="wrist_mount_visual", type=int(mujoco.mjtGeom.mjGEOM_MESH), meshname="wrist_mount",
        pos=[0, 0, -BRACKET_FLANGE_Z_MM / 1000], rgba=[0.15, 0.15, 0.17, 1.0],
        contype=0, conaffinity=0, density=0, group=1)
    tool.add_geom(
        name="wrist_camera_visual", type=int(mujoco.mjtGeom.mjGEOM_BOX),
        pos=list(CAMERA_HOUSING_POS), quat=list(CAMERA_HOUSING_QUAT), size=list(CAMERA_HOUSING_HALF),
        rgba=[0.08, 0.08, 0.09, 1.0], contype=0, conaffinity=0, density=0, group=1)
    # Ø90 mm puck from the bracket's back face (tool0 z = -6.4 mm) for 24.4 mm.
    tool.add_geom(
        name="wrist_mount_puck_collision", type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        pos=[0, 0, 0.0058], size=[0.045, 0.0122, 0], **_MOUNT_COLLISION)
    # Ø22 mm tube along tool0 +y from the puck rim to the camera plate.
    tool.add_geom(
        name="wrist_mount_tube_collision", type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        fromto=[0, 0.045, 0.0056, 0, 0.113, 0.0056], size=[0.011, 0, 0], **_MOUNT_COLLISION)
    # Camera plate block under the tilted face.
    tool.add_geom(
        name="wrist_mount_plate_collision", type=int(mujoco.mjtGeom.mjGEOM_BOX),
        pos=[0, 0.12465, 0.0136], size=[0.045, 0.01175, 0.019], **_MOUNT_COLLISION)
    tool.add_geom(
        name="wrist_camera_collision", type=int(mujoco.mjtGeom.mjGEOM_BOX),
        pos=list(CAMERA_HOUSING_POS), quat=list(CAMERA_HOUSING_QUAT), size=list(CAMERA_HOUSING_HALF),
        **_MOUNT_COLLISION)
    tool.add_camera(
        name=WRIST_CAMERA, pos=list(WRIST_CAMERA_POS), xyaxes=list(WRIST_CAMERA_XYAXES),
        fovy=WRIST_CAMERA_FOVY)
    for other in (*ARM_BODIES[:6], *(b.name for b in spec.bodies if b.name.startswith(GRIPPER_PREFIX))):
        spec.add_exclude(bodyname1=TOOL_BODY, bodyname2=other)



def _use_legacy_cup_visuals(spec: mujoco.MjSpec) -> None:
    """Restore the original STL shells and shadows for the cup demo camera."""
    for body_name in ARM_MESH_BODIES:
        body = spec.body(body_name)
        geoms = list(body.geoms)
        shell = next(geom for geom in geoms if geom.group == 3 and geom.meshname)
        visuals = [geom for geom in geoms if geom.group == 1 and geom.meshname]
        first, *remaining = visuals
        first.meshname = shell.meshname
        first.rgba = [0.5, 0.5, 0.5, 1.0]
        for geom in remaining:
            spec.delete(geom)
        shell.group = 0
    for mesh in list(spec.meshes):
        if "_visual_" in mesh.name:
            spec.delete(mesh)


def _apply_cabinet(spec: mujoco.MjSpec) -> None:
    """Gravity compensation + PD that can hold and track without sagging into the cups."""
    for name in ARM_BODIES:
        spec.body(name).gravcomp = 1.0
    gains = dict(zip(ARM_JOINTS, zip(ARM_KP, ARM_KV)))
    for joint in spec.joints:
        if joint.name in gains:
            joint.damping = [0.0, 0.0, 0.0]  # MuJoCo 4: [linear, quadratic, cubic]
            joint.armature = 0.1
    for act in spec.actuators:
        pair = gains.get(act.target)
        if pair is None:
            continue
        kp, kv = pair
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv


def _apply_arm_collisions(spec: mujoco.MjSpec) -> None:
    """Official UR STLs overlap neighbours and the 2F-140; keep them visual, collide as capsules."""
    for body_name in ARM_MESH_BODIES:
        for geom in spec.body(body_name).geoms:
            geom.contype = 0
            geom.conaffinity = 0
            geom.density = 0
    capsule = dict(
        type=int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        contype=1,
        conaffinity=1,
        condim=3,
        density=0,
        group=3,
        friction=[0.5, 0.005, 0.0001],
    )
    spec.body("shoulder_link").add_geom(
        name="shoulder_collision", pos=[0.0, 0.0, 0.0], size=[0.078, 0.07, 0.0], **capsule)
    # Tube runs along -x at the official visual offset z=0.1762; inset so joint capsules do not weld.
    spec.body("upper_arm_link").add_geom(
        name="upper_arm_joint_collision", pos=[0.0, 0.0, 0.08], size=[0.075, 0.07, 0.0], **capsule)
    spec.body("upper_arm_link").add_geom(
        name="upper_arm_collision", fromto=[-0.08, 0.0, 0.1762, -0.54, 0.0, 0.1762],
        size=[0.065, 0.0, 0.0], **capsule)
    spec.body("forearm_link").add_geom(
        name="forearm_collision", fromto=[-0.08, 0.0, 0.04, -0.48, 0.0, 0.04],
        size=[0.048, 0.0, 0.0], **capsule)
    spec.body("wrist_1_link").add_geom(
        name="wrist_1_collision", fromto=[0.0, -0.02, 0.0, 0.0, -0.10, 0.0],
        size=[0.045, 0.0, 0.0], **capsule)
    spec.body("wrist_2_link").add_geom(
        name="wrist_2_collision", fromto=[0.0, 0.02, 0.0, 0.0, 0.08, 0.0],
        size=[0.042, 0.0, 0.0], **capsule)
    # Approximate the wrist-3 shell by its 90 mm diameter and 50 mm axial
    # envelope, fitted to the vendored UR collision mesh in this link frame.
    spec.body("wrist_3_link").add_geom(
        name="wrist_3_collision", type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        pos=[0, 0, -0.025], size=[0.045, 0.025, 0],
        contype=1, conaffinity=1, condim=3, density=0, group=3,
        friction=[0.5, 0.005, 0.0001])
