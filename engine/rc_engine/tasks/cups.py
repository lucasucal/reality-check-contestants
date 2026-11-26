"""Plastic-cup task family on the UR12e + Robotiq 2F-140 cell.

One env class, five public task kinds (lift / pick / place / tip / stack). A task is data:
`CupParams` fully describes layout ranges, cup size and episode length, so hidden tasks
ship as JSON packs (rc_engine/packs.py) instead of code and never appear in the public
image. The scene is composed with MjSpec on top of the vendored UR12e and 2F-140 (SO-101
assets stay on disk and are not loaded). 20 Hz `joint_delta` is interpolated
across 5 ms physics with UR's 0.1 s servoj lookahead.

World frame: UR12e base at the origin on the table (z=0), +x forward, +y left. Layouts are
sampled in polar coordinates around the base because that is the arm's natural workspace.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any

import mujoco
import numpy as np

from ..embodiment import (
    ARM_KP,
    ARM_KV,
    CUP_R,
    GRIPPER_ACTUATOR,
    GRIPPER_JOINT,
    GRIPPERFRAME,
    HOME_Q,
    PHYSICS_DT,
    SERVOJ_LOOKAHEAD,
    TARGET_R,
    TOP_CAMERA_FOVY,
    TOP_CAMERA_POS,
    TOP_CAMERA_XYAXES,
    robot_spec,
)
from ..env import StepResult, hash_state
from ..schema import NUM_JOINTS, CameraSpec, JointDelta, RawObs, TaskSpec

SUBSTEPS = 10  # x PHYSICS_DT = 20 Hz control; inner loop interpolates like servoj
CONTROL_HZ = 20
CONTROL_DT = SUBSTEPS * PHYSICS_DT
_LOOKAHEAD_ALPHA = PHYSICS_DT / SERVOJ_LOOKAHEAD
_ARM_KP = np.array(ARM_KP)
_ARM_KV = np.array(ARM_KV)
CAMERAS = (CameraSpec("top", 640, 480),)  # world overhead only; live wrist may be null
MJ_CAMERA = {"top": "top"}
KINDS = ("lift", "pick", "place", "tip", "stack")

_VIEWER = None

UPRIGHT_Z = 0.94  # cos(20 deg): cup axis vs world z
TIPPED_Z = 0.34  # cos(70 deg)
REST_SPEED = 0.02
LIFT_HEIGHT = 0.04  # centre-of-cup rise that counts as "lifted"
LIFT_HOLD_STEPS = 10
TIP_HOLD_STEPS = 5
GRIPPER_OPEN = 0.5
DISTRACTOR_MOVED = 0.02  # pick task: the blue cup must stay put


@dataclass(frozen=True)
class CupParams:
    kind: str
    description: str = ""
    episode_length: int = 300
    cup_height: float = 0.06
    cup_r_top: float = 0.0275
    cup_r_bottom: float = 0.021
    cup_mass: float = 0.015
    cup_friction: float = 1.0  # MuJoCo takes the max of a pair: cup<->table is max(cup, 0.1)
    # polar sampling ranges around the base: (min, max)
    cup_r: tuple[float, float] = CUP_R
    cup_theta: tuple[float, float] = (-0.6, 0.6)
    target_r: tuple[float, float] = TARGET_R
    target_theta: tuple[float, float] = (-0.6, 0.6)
    min_separation: float = 0.09  # cup <-> target/other cup, centre to centre
    success_xy: float = 0.03
    target_radius: float = 0.035

    @classmethod
    def from_json(cls, kind: str, raw: dict[str, Any]) -> "CupParams":
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - names)
        if unknown or kind not in KINDS:
            raise ValueError(f"bad cup task: kind={kind!r}, unknown params {unknown}")
        pair = {"cup_r", "cup_theta", "target_r", "target_theta"}
        clean = {k: (tuple(float(x) for x in v) if k in pair else v) for k, v in raw.items()}
        return replace(cls(kind=kind), **clean)


# The five public practice tasks (the same ids teams see in the public engine image).
PUBLIC: dict[str, CupParams] = {
    "cup_lift_v0": CupParams(
        "lift", "Grasp the cup and lift it at least 4 cm off the table; hold it for half a second."),
    "cup_pick_red_v0": CupParams(
        "pick", "Lift the red cup at least 4 cm and hold it, without disturbing the blue cup.",
        cup_theta=(-0.6, -0.1), target_theta=(0.1, 0.6)),
    "cup_place_v0": CupParams(
        "place", "Pick up the cup, set it upright on the green target circle and release it.",
        cup_theta=(-0.6, -0.1), target_theta=(0.1, 0.6)),
    "cup_tip_v0": CupParams(
        "tip", "Knock the upright cup over so it lies on its side."),
    "cup_stack_v0": CupParams(
        "stack", "Pick up the red cup and set it on top of the blue cup, then release.",
        cup_theta=(-0.6, -0.1), target_theta=(0.1, 0.6)),
}


def _frustum(n: int, r_bottom: float, r_top: float, h: float) -> list[float]:
    verts = []
    for z, r in ((-h / 2, r_bottom), (h / 2, r_top)):
        for i in range(n):
            a = 2 * np.pi * i / n
            verts += [r * np.cos(a), r * np.sin(a), z]
    return verts


def _build(p: CupParams) -> mujoco.MjModel:
    # Preserve the legacy demo policy's calibrated camera appearance.
    spec = robot_spec(legacy_cup_visuals=True)
    spec.option.timestep = PHYSICS_DT
    # tabletop + lighting (the vendored arm+gripper have neither)
    spec.add_texture(name="tabletex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.78, 0.76, 0.72], width=16, height=16)
    mat = spec.add_material(name="table")
    mat.rgba = [0.78, 0.76, 0.72, 1]
    wb = spec.worldbody
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.05], material="table", friction=[0.1, 0.005, 0.0001])
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0, 0, 0]
    wb.add_light(pos=[0.5, 0, 1.8], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05], castshadow=False)
    # fixed overhead camera: image-up = +x (away from the base), image-right = -y
    wb.add_camera(name="top", pos=list(TOP_CAMERA_POS), xyaxes=list(TOP_CAMERA_XYAXES), fovy=TOP_CAMERA_FOVY)
    spec.add_mesh(name="cup", uservert=_frustum(24, p.cup_r_bottom, p.cup_r_top, p.cup_height))

    def cup(name: str, rgba: list[float]) -> None:
        body = wb.add_body(name=name, pos=[0.0, 0.0, p.cup_height / 2])
        body.add_freejoint(name=f"{name}_free")
        body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cup", rgba=rgba, mass=p.cup_mass,
                      friction=[p.cup_friction, 0.005, 0.0001], condim=4, solref=[0.02, 1])

    cup("cup", [0.85, 0.12, 0.12, 1])
    if p.kind in ("stack", "pick"):
        cup("cup2", [0.15, 0.3, 0.85, 1])
    if p.kind == "place":
        marker = wb.add_body(name="target", pos=[0.0, 0.0, 0.0005], mocap=True)
        marker.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p.target_radius, 0.0005, 0], rgba=[0.1, 0.75, 0.25, 1],
                        contype=0, conaffinity=0)
    return spec.compile()


def _polar(rng: np.random.Generator, r: tuple[float, float], theta: tuple[float, float]) -> np.ndarray:
    rad, th = rng.uniform(*r), rng.uniform(*theta)
    return np.array([rad * np.cos(th), rad * np.sin(th)])


class LayoutError(RuntimeError):
    pass


def sample_layout(p: CupParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(cup xy, second object xy: target circle / other cup) for one seed; deterministic in rng."""
    cup_xy = _polar(rng, p.cup_r, p.cup_theta)
    for _ in range(1000):
        other = _polar(rng, p.target_r, p.target_theta)
        if np.linalg.norm(other - cup_xy) >= p.min_separation:
            return cup_xy, other
    raise LayoutError(f"no layout with separation {p.min_separation} for cup_r={p.cup_r} cup_theta={p.cup_theta} "
                      f"target_r={p.target_r} target_theta={p.target_theta}")


def check_layouts(p: CupParams, seeds: range = range(200)) -> None:
    """Fail at pack-load time, not mid-run, if any seed has no valid layout."""
    for seed in seeds:
        sample_layout(p, np.random.Generator(np.random.PCG64(seed)))


def _sync_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Organizer-only MuJoCo window. Off unless RC_ENGINE_VIEWER=1 and DISPLAY is set."""
    global _VIEWER
    if os.environ.get("RC_ENGINE_VIEWER") != "1" or not os.environ.get("DISPLAY"):
        return
    if _VIEWER is None:
        from mujoco import viewer as mj_viewer
        _VIEWER = mj_viewer.launch_passive(model, data)
    elif _VIEWER.is_running():
        _VIEWER.sync()


class CupsEnv:
    def __init__(self, task_id: str, params: CupParams, visibility: str = "public") -> None:
        self.params = params
        self.spec = TaskSpec(task_id, params.description, params.episode_length, CONTROL_HZ, CAMERAS,
                             family="cups", visibility=visibility)
        self.model = _build(params)
        m = self.model
        self.data = mujoco.MjData(m)
        self.ctrl_range = m.actuator_ctrlrange.copy()
        self.tip = m.site(GRIPPERFRAME).id
        self.cup_qpos = m.jnt_qposadr[m.joint("cup_free").id]
        self.cup_qvel = m.jnt_dofadr[m.joint("cup_free").id]
        self.cup_body = m.body("cup").id
        self.cup2_body = m.body("cup2").id if params.kind in ("stack", "pick") else None
        self.cup2_qpos = m.jnt_qposadr[m.joint("cup2_free").id] if params.kind in ("stack", "pick") else None
        self.target_mocap = m.body_mocapid[m.body("target").id] if params.kind == "place" else None
        self.state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.renderer: mujoco.Renderer | None = None
        self.target = np.array(HOME_Q, dtype=float)
        self.q_cmd = np.array(HOME_Q, dtype=float)
        self.step_count = 0
        self.lift_hold = 0
        self.tip_hold = 0
        self.cup2_start = np.zeros(2)
        grip_jnt = m.joint(GRIPPER_JOINT).id
        self.grip_act = m.actuator(GRIPPER_ACTUATOR).id
        self.grip_qadr = int(m.jnt_qposadr[grip_jnt])
        closed_q, open_q = (float(x) for x in m.jnt_range[grip_jnt])
        self.grip_closed_q, self.grip_open_q = closed_q, open_q  # 2F-140: -0.8 closed .. 0 open
        open_ctrl, closed_ctrl = (float(x) for x in m.actuator_ctrlrange[self.grip_act])
        self.grip_open_ctrl, self.grip_closed_ctrl = open_ctrl, closed_ctrl  # 0 open .. 255 closed
        self.render = True  # tests that only need physics switch this off

    # --- Env protocol -------------------------------------------------------------
    def reset(self, seed: int) -> RawObs:
        m, d, p = self.model, self.data, self.params
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(m, d)
        d.qpos[:NUM_JOINTS] = HOME_Q
        d.qpos[self.grip_qadr] = self.grip_open_q
        d.ctrl[:NUM_JOINTS] = HOME_Q
        d.ctrl[self.grip_act] = self.grip_open_ctrl
        self.q_cmd = np.array(HOME_Q, dtype=float)
        cup_xy, other = sample_layout(p, rng)
        self._place_cup(self.cup_qpos, cup_xy, p.cup_height / 2)
        if self.cup2_qpos is not None:
            self._place_cup(self.cup2_qpos, other, p.cup_height / 2)
            self.cup2_start = other.copy()
        if self.target_mocap is not None:
            d.mocap_pos[self.target_mocap] = [*other, 0.0005]
        self.target = np.array(HOME_Q, dtype=float)
        self.step_count = 0
        self.lift_hold = 0
        self.tip_hold = 0
        mujoco.mj_forward(m, d)
        _sync_viewer(m, d)
        return self._obs()

    def step(self, action: JointDelta) -> StepResult:
        d = self.data
        self.target = np.clip(self.target + action.values, self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
        qd_des = np.asarray(action.values, dtype=float) / CONTROL_DT
        d.ctrl[self.grip_act] = self.grip_closed_ctrl + action.gripper * (self.grip_open_ctrl - self.grip_closed_ctrl)
        # 20 Hz host stream → 5 ms servoj: first-order lookahead + velocity feedforward.
        for _ in range(SUBSTEPS):
            self.q_cmd += _LOOKAHEAD_ALPHA * (self.target - self.q_cmd)
            d.ctrl[:NUM_JOINTS] = np.clip(
                self.q_cmd + (_ARM_KV / _ARM_KP) * qd_des,
                self.ctrl_range[:NUM_JOINTS, 0], self.ctrl_range[:NUM_JOINTS, 1])
            mujoco.mj_step(self.model, d)
        _sync_viewer(self.model, d)
        self.step_count += 1
        obs = self._obs()
        success = self._success(obs.gripper)
        return StepResult(obs, success or self.step_count >= self.spec.episode_length, success)

    def state_hash(self) -> str:
        mujoco.mj_getState(self.model, self.data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return hash_state(self.step_count, self.state, self.q_cmd)

    # --- layout -------------------------------------------------------------------
    def _place_cup(self, qpos: int, xy: np.ndarray, z: float) -> None:
        self.data.qpos[qpos:qpos + 7] = [*xy, z, 1, 0, 0, 0]

    # --- success predicates -------------------------------------------------------
    def _cup_axis_z(self, body: int) -> float:
        return float(self.data.xmat[body].reshape(3, 3)[2, 2])

    def _cup_at_rest(self) -> bool:
        # linear speed only: hull-mesh contacts leave a harmless angular buzz on a resting cup
        return bool(np.linalg.norm(self.data.qvel[self.cup_qvel:self.cup_qvel + 3]) < REST_SPEED)

    def _success(self, gripper: float) -> bool:
        d, p = self.data, self.params
        cup = d.xpos[self.cup_body]
        upright = self._cup_axis_z(self.cup_body) > UPRIGHT_Z
        if p.kind == "tip":
            self.tip_hold = self.tip_hold + 1 if self._cup_axis_z(self.cup_body) < TIPPED_Z else 0
            return self.tip_hold >= TIP_HOLD_STEPS
        if p.kind in ("lift", "pick"):
            held = cup[2] > p.cup_height / 2 + LIFT_HEIGHT and self._cup_axis_z(self.cup_body) > 0.8
            if p.kind == "pick" and np.linalg.norm(d.xpos[self.cup2_body][:2] - self.cup2_start) > DISTRACTOR_MOVED:
                held = False
            self.lift_hold = self.lift_hold + 1 if held else 0
            return self.lift_hold >= LIFT_HOLD_STEPS
        released = gripper > GRIPPER_OPEN and upright and self._cup_at_rest()
        if p.kind == "place":
            return bool(np.linalg.norm(cup[:2] - d.mocap_pos[self.target_mocap][:2]) < p.success_xy and released)
        base = d.xpos[self.cup2_body]  # stack
        on_top = np.linalg.norm(cup[:2] - base[:2]) < 0.02 and abs(cup[2] - (base[2] + p.cup_height)) < 0.012
        return bool(on_top and released and self._cup_axis_z(self.cup2_body) > UPRIGHT_Z)

    # --- observation --------------------------------------------------------------
    def _obs(self) -> RawObs:
        m, d = self.model, self.data
        images = {}
        if self.render:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(m, CAMERAS[0].height, CAMERAS[0].width)
            for cam in CAMERAS:
                self.renderer.update_scene(d, camera=MJ_CAMERA[cam.name])
                images[cam.name] = self.renderer.render()
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
