"""Five easier lifts of robosuite's Lift, on that repo's table and object meshes.

Robosuite Lift picks up one object. The only permutation is which of its PickPlace
meshes is on the table (can, cereal, bread, milk, bottle), spawned in a tight patch
in front of the UR12e so a policy can be trained on overhead images. The policy
observation is still the single `top` camera. `booth` and `side` are rendered only
when RC_AUDIENCE_CAMERAS=1, and the engine puts them on the live-view channel.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..embodiment import (
    GRIPPER_ACTUATOR,
    GRIPPER_JOINT,
    GRIPPERFRAME,
    HOME_Q,
    PHYSICS_DT,
    TOP_CAMERA_FOVY,
    TOP_CAMERA_POS,
    TOP_CAMERA_XYAXES,
    robot_spec,
)
from ..env import StepResult, hash_state
from ..schema import NUM_JOINTS, JointDelta, RawObs, TaskSpec
from .cups import (
    CAMERAS,
    CONTROL_DT,
    CONTROL_HZ,
    LIFT_HEIGHT,
    LIFT_HOLD_STEPS,
    SUBSTEPS,
    _ARM_KP,
    _ARM_KV,
    _LOOKAHEAD_ALPHA,
)

_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "robosuite"
_OBJECTS = _ASSETS / "objects"
_TEXTURES = _ASSETS / "textures"
AUDIENCE = ("booth", "side")


@dataclass(frozen=True)
class LiftObject:
    key: str
    label: str
    bottom: float  # site z in the object root, from the robosuite XML
    top: float

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def spawn_z(self) -> float:
        return -self.bottom

    @property
    def grasp_z(self) -> float:
        # Palm height that puts the 2F-140 fingers on a cup also works here:
        # cups use palm z=0.122 with the centre at 0.03.
        return self.height / 2 + 0.092

    @property
    def hover_z(self) -> float:
        return self.grasp_z + 0.08

    @property
    def lift_z(self) -> float:
        return self.grasp_z + 0.11


OBJECTS: dict[str, LiftObject] = {
    "can": LiftObject("can", "soda can", -0.06, 0.04),
    "cereal": LiftObject("cereal", "cereal box", -0.10, 0.03),
    "bread": LiftObject("bread", "loaf of bread", -0.045, 0.03),
    "milk": LiftObject("milk", "milk carton", -0.085, 0.075),
    "bottle": LiftObject("bottle", "bottle", -0.082, 0.075),
}

TASKS: dict[str, LiftObject] = {f"bench_lift_{key}_v0": obj for key, obj in OBJECTS.items()}


def _look(eye: tuple[float, float, float], target: tuple[float, float, float]) -> list[float]:
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def _material(spec: mujoco.MjSpec, name: str, texture: str, *, specular: float, shininess: float,
              reflectance: float, repeat: tuple[float, float]) -> None:
    mat = spec.add_material(name=name)
    mat.specular = specular
    mat.shininess = shininess
    mat.reflectance = reflectance
    mat.texrepeat = list(repeat)
    mat.texuniform = True
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture


def _arena(spec: mujoco.MjSpec) -> None:
    """robosuite table_arena.xml, sat on z=0 so the UR12e base stays where it is."""
    tex = spec.add_texture
    cube = mujoco.mjtTexture.mjTEXTURE_CUBE
    image = mujoco.mjtTexture.mjTEXTURE_2D
    tex(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.9, 0.9, 1.0], rgb2=[0.2, 0.3, 0.4], width=256, height=256)
    tex(name="texplane", type=image, file=str(_TEXTURES / "light-gray-floor-tile.png"))
    tex(name="arena-ceramic", type=cube, file=str(_TEXTURES / "ceramic.png"))
    tex(name="arena-steel", type=cube, file=str(_TEXTURES / "steel-brushed.png"))
    tex(name="arena-plaster", type=image, file=str(_TEXTURES / "light-gray-plaster.png"))
    _material(spec, "floorplane", "texplane", specular=0.0, shininess=0.0, reflectance=0.01, repeat=(2, 2))
    _material(spec, "table_ceramic", "arena-ceramic", specular=0.2, shininess=0.0, reflectance=0.0, repeat=(1, 1))
    _material(spec, "table_legs_metal", "arena-steel", specular=0.5, shininess=0.8, reflectance=0.8, repeat=(1, 1))
    _material(spec, "walls_mat", "arena-plaster", specular=0.1, shininess=0.1, reflectance=0.0, repeat=(3, 3))
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.38]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    wb = spec.worldbody
    # Hidden collider. The visible top is the ceramic box; objects still rest at z=0.
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2.0, 2.0, 0.05], group=3,
                rgba=[0.0, 0.0, 0.0, 0.0], friction=[1.0, 0.005, 0.0001])
    cx, cy, hx, hy, hz = 0.48, 0.0, 0.72, 0.52, 0.025
    wb.add_geom(name="table_visual", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx, cy, -hz],
                size=[hx, hy, hz], material="table_ceramic", contype=0, conaffinity=0)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0.4, 0.0, -0.75],
                size=[3.0, 3.0, 0.125], material="floorplane", contype=0, conaffinity=0)
    leg_z, leg_half = -0.40, 0.35
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        wb.add_geom(name=f"table_leg{i}_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[cx + sx * (hx - 0.08), cy + sy * (hy - 0.08), leg_z],
                    size=[0.025, leg_half, 0.0], material="table_legs_metal", contype=0, conaffinity=0)
    wb.add_geom(name="wall_rear_visual", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.55, 0.0, 0.55],
                size=[0.02, 1.6, 1.3], material="walls_mat", contype=0, conaffinity=0)
    wb.add_geom(name="wall_left_visual", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.5, 1.85, 0.55],
                size=[1.4, 0.02, 1.3], material="walls_mat", contype=0, conaffinity=0)
    # Same light as table_arena.xml, with the shadow map turned on.
    wb.add_light(pos=[1.0, 1.0, 1.5], dir=[-0.2, -0.2, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=True, specular=[0.3, 0.3, 0.3], diffuse=[0.7, 0.7, 0.7])
    wb.add_camera(name="top", pos=list(TOP_CAMERA_POS), xyaxes=list(TOP_CAMERA_XYAXES), fovy=TOP_CAMERA_FOVY)
    focus = (0.55, 0.0, 0.2)
    booth = (1.75, -1.35, 1.15)
    side = (0.30, 1.15, 0.78)
    wb.add_camera(name="booth", pos=list(booth), xyaxes=_look(booth, focus), fovy=48)
    wb.add_camera(name="side", pos=list(side), xyaxes=_look(side, (0.75, 0.05, 0.12)), fovy=40)


# The cereal box and the bottle rock on the convex hull of their meshes.
# A flat primitive carries the contact; the mesh stays the thing you see.
_COLLISION = {
    "cereal": ("box", [0.0, 0.0, 0.0003], [0.022, 0.0325, 0.0315]),
    "bottle": ("cylinder", [0.0, 0.0, 0.005], [0.026, 0.070, 0.0]),
}


def _attach_object(spec: mujoco.MjSpec, obj: LiftObject) -> None:
    child = mujoco.MjSpec.from_file(str(_OBJECTS / f"{obj.key}.xml"))
    root = child.worldbody.bodies[0]
    root.name = "root"
    root.add_freejoint(name="free")
    primitive = _COLLISION.get(obj.key)
    if primitive is not None:
        kind, pos, size = primitive
        body = root.bodies[0]
        for geom in body.geoms:
            geom.contype = 0
            geom.conaffinity = 0
            geom.density = 0
        body.add_geom(
            name="grasp",
            type=mujoco.mjtGeom.mjGEOM_BOX if kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos, size=size, group=3, rgba=[0.0, 0.0, 0.0, 0.0],
            friction=[1.1, 0.04, 0.001], density=300, condim=4,
        )
    spec.attach(child, prefix=f"{obj.key}/", frame=spec.worldbody.add_frame())


def _build(obj: LiftObject) -> mujoco.MjModel:
    spec = robot_spec()
    spec.option.timestep = PHYSICS_DT
    _arena(spec)
    _attach_object(spec, obj)
    return spec.compile()


def _rest_z(obj: LiftObject) -> float:
    """Height of the free joint so the contact shape starts on the table."""
    primitive = _COLLISION.get(obj.key)
    if primitive is None:
        return obj.spawn_z
    kind, pos, size = primitive
    half_z = size[2] if kind == "box" else size[1]
    return -(pos[2] - half_z)


def _spawn_xy(rng: np.random.Generator) -> np.ndarray:
    # Tight patch in front of the gripper. Wider spawns knock the boxy objects over.
    radius, theta = rng.uniform(0.824, 0.840), rng.uniform(-0.035, 0.035)
    return np.array([radius * np.cos(theta), radius * np.sin(theta)])


class BenchEnv:
    def __init__(self, task_id: str, obj: LiftObject, visibility: str = "public") -> None:
        self.obj = obj
        self.params = type("LiftParams", (), {"kind": "lift", "cup_height": obj.height})()
        self.spec = TaskSpec(
            task_id, f"Lift the {obj.label} at least 4 cm off the table and hold it.",
            300, CONTROL_HZ, CAMERAS, family="bench", visibility=visibility)
        self.model = _build(obj)
        m = self.model
        self.data = mujoco.MjData(m)
        self.ctrl_range = m.actuator_ctrlrange.copy()
        self.tip = m.site(GRIPPERFRAME).id
        self.cup_body = m.body(f"{obj.key}/root").id
        self.free_qpos = int(m.jnt_qposadr[m.joint(f"{obj.key}/free").id])
        self.grasp_z = obj.grasp_z
        self.hover_z = obj.hover_z
        self.lift_z = obj.lift_z
        self.state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.renderer: mujoco.Renderer | None = None
        self.audience = os.environ.get("RC_AUDIENCE_CAMERAS") == "1"
        self.audience_images: dict[str, np.ndarray] = {}
        self.target = np.array(HOME_Q, dtype=float)
        self.q_cmd = np.array(HOME_Q, dtype=float)
        self.step_count = 0
        self.lift_hold = 0
        grip_jnt = m.joint(GRIPPER_JOINT).id
        self.grip_act = m.actuator(GRIPPER_ACTUATOR).id
        self.grip_qadr = int(m.jnt_qposadr[grip_jnt])
        closed_q, open_q = (float(x) for x in m.jnt_range[grip_jnt])
        self.grip_closed_q, self.grip_open_q = closed_q, open_q
        open_ctrl, closed_ctrl = (float(x) for x in m.actuator_ctrlrange[self.grip_act])
        self.grip_open_ctrl, self.grip_closed_ctrl = open_ctrl, closed_ctrl
        self.render = True

    def reset(self, seed: int) -> RawObs:
        m, d = self.model, self.data
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(m, d)
        d.qpos[:NUM_JOINTS] = HOME_Q
        d.qpos[self.grip_qadr] = self.grip_open_q
        d.ctrl[:NUM_JOINTS] = HOME_Q
        d.ctrl[self.grip_act] = self.grip_open_ctrl
        self.q_cmd = np.array(HOME_Q, dtype=float)
        xy = _spawn_xy(rng)
        d.qpos[self.free_qpos:self.free_qpos + 7] = [*xy, _rest_z(self.obj), 1, 0, 0, 0]
        self.target = np.array(HOME_Q, dtype=float)
        self.step_count = 0
        self.lift_hold = 0
        mujoco.mj_forward(m, d)
        for _ in range(40):  # let the mesh settle on the ceramic top before the episode starts
            mujoco.mj_step(m, d)
        self.rest_z = float(d.xpos[self.cup_body][2])
        # Palm site sits ~9 cm above the pinch. 0.11 clears the table and closes on the body.
        self.grasp_z = self.rest_z + 0.11
        self.hover_z = self.grasp_z + 0.12
        self.lift_z = self.grasp_z + 0.12
        return self._obs()

    def step(self, action: JointDelta) -> StepResult:
        d = self.data
        self.target = np.clip(self.target + action.values, self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
        qd_des = np.asarray(action.values, dtype=float) / CONTROL_DT
        d.ctrl[self.grip_act] = self.grip_closed_ctrl + action.gripper * (self.grip_open_ctrl - self.grip_closed_ctrl)
        for _ in range(SUBSTEPS):
            self.q_cmd += _LOOKAHEAD_ALPHA * (self.target - self.q_cmd)
            d.ctrl[:NUM_JOINTS] = np.clip(
                self.q_cmd + (_ARM_KV / _ARM_KP) * qd_des,
                self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
            mujoco.mj_step(self.model, d)
        self.step_count += 1
        obs = self._obs()
        held = self.data.xpos[self.cup_body][2] > self.rest_z + LIFT_HEIGHT
        self.lift_hold = self.lift_hold + 1 if held else 0
        success = self.lift_hold >= LIFT_HOLD_STEPS
        return StepResult(obs, success or self.step_count >= self.spec.episode_length, success)

    def state_hash(self) -> str:
        mujoco.mj_getState(self.model, self.data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return hash_state(self.step_count, self.state, self.q_cmd)

    def _grab(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        assert self.renderer is not None
        self.renderer.update_scene(data, camera=camera)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        return self.renderer.render()

    def _obs(self) -> RawObs:
        m, d = self.model, self.data
        images: dict[str, np.ndarray] = {}
        self.audience_images = {}
        if self.render:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(m, CAMERAS[0].height, CAMERAS[0].width)
            names = [cam.name for cam in CAMERAS]
            if self.audience:
                names.extend(AUDIENCE)
            rendered = {name: self._grab(d, name) for name in names}
            images = {cam.name: rendered[cam.name] for cam in CAMERAS}
            if self.audience:
                self.audience_images = {name: rendered[name] for name in AUDIENCE}
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.tip])
        return RawObs(
            images=images,
            joint_pos=d.qpos[:NUM_JOINTS].copy(),
            joint_vel=d.qvel[:NUM_JOINTS].copy(),
            gripper=float(np.clip(
                (d.qpos[self.grip_qadr] - self.grip_closed_q) / (self.grip_open_q - self.grip_closed_q), 0.0, 1.0)),
            ee_pos=d.site_xpos[self.tip].copy(),
            ee_quat=quat,
        )
