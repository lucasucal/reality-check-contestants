"""UR12e motion and safety limits the engine applies or measures (contestants/RULES.md "Action boundary").

The numbers are the UR12e's published joint speeds and its default (Normal) safety-mode limits:

- Each policy increment is clamped to ±0.05 rad per 50 ms control step (schema.MAX_JOINT_DELTA).
- TCP and elbow speed, 1.5 m/s each (`scale_step`): the commanded speed of a step is |J_pos(q_target) · Δq| / dt,
  with J_pos the MuJoCo position Jacobian of the TCP (site `gripperframe`) or the elbow (origin of body
  `forearm_link`) at the commanded joint target, Δq the step's joint-target change after the per-joint clamp and
  dt one control step. When the faster of the two is above 1.5 m/s the whole Δq is scaled down by one factor
  (direction kept) so that the faster of the two, with the same Jacobians, is exactly 1.5 m/s
  (hardware/bridge.py applies the same rule on the arm).
- TCP force 150 N, momentum 25 kg·m/s, power 300 W (`SafetyMeter`): measured, never enforced. Each is sampled
  every physics substep and averaged over the control step (50 ms); a repetition's peak is the largest of these
  control-step means. The UR estimates TCP force from filtered joint torques, so a single-substep contact impulse
  (a part landing in a seat) is not what its safety system sees. A repetition whose peak goes over one of them
  would trigger a safety stop on the real arm; the runner shows a notice on public tasks. Not part of the score.
- Protective stop (`safety_stop_mode`, opt-in): with RC_SAFETY_STOP=enforce the engine also acts on that same
  measurement. The first control step whose mean goes over one of the three limits ends the episode,
  unsuccessful, with `/step` `info.protective_stop` {limit, value, step}; the step is simulated in full (the mean
  needs all of it). Same samples, same averaging, same thresholds: no second force check. Off by default
  (RC_SAFETY_STOP unset or "measure"), so default and judged runs measure only. `/readyz` `build.safety_stop`
  records the mode.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from .embodiment import ARM_JOINTS, GRIPPER_PREFIX, GRIPPERFRAME
from .schema import ELBOW_SPEED_LIMIT, TCP_SPEED_LIMIT
ELBOW_BODY = "forearm_link"  # its origin is the elbow joint
JOINT_STEP_CLAMP = 0.05  # rad per control step, every joint (the real arm's driver)
TCP_FORCE_LIMIT = 150.0  # N
MOMENTUM_LIMIT = 25.0  # kg m/s
POWER_LIMIT = 300.0  # W
SAFETY_LIMITS = {"tcp_force_n": TCP_FORCE_LIMIT, "momentum_kg_m_s": MOMENTUM_LIMIT, "power_w": POWER_LIMIT}
SAFETY_STOP_ENV = "RC_SAFETY_STOP"
SAFETY_STOP_MODES = ("measure", "enforce")


def safety_stop_mode() -> str:
    """"measure" (default: RC_SAFETY_STOP unset, empty or "measure") or "enforce". Anything else is a
    configuration error, raised rather than guessed, so a typo never silently changes how an episode can end."""
    raw = os.environ.get(SAFETY_STOP_ENV, "").strip()
    mode = raw.lower() or "measure"
    if mode not in SAFETY_STOP_MODES:
        raise ValueError(f"{SAFETY_STOP_ENV} must be one of {', '.join(SAFETY_STOP_MODES)} (or unset), got {raw!r}")
    return mode


def over_limit(means: dict[str, float]) -> tuple[str, float] | None:
    """(key, value) of the first SAFETY_LIMITS key whose control-step mean is over its limit, else None."""
    for key, limit in SAFETY_LIMITS.items():
        value = float(means.get(key, 0.0))
        if value > limit:
            return key, value
    return None


class SpeedScaler:
    """TCP and elbow speed limit of one model (a scratch MjData evaluates the Jacobians at the joint target)."""

    def __init__(self, model: mujoco.MjModel, dt: float) -> None:
        self.model, self.dt = model, float(dt)
        self.data = mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
        self.dofs = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.site = model.site(GRIPPERFRAME).id
        self.elbow = model.body(ELBOW_BODY).id
        self._jt = np.zeros((3, model.nv))
        self._je = np.zeros((3, model.nv))

    def speeds(self, target: np.ndarray, dq: np.ndarray) -> tuple[float, float]:
        """Commanded (TCP, elbow) speed in m/s of joint-target change `dq` ending at joint target `target`."""
        m, d = self.model, self.data
        d.qpos[self.qadr] = target
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, self._jt, None, self.site)
        mujoco.mj_jacBody(m, d, self._je, None, self.elbow)
        dq = np.asarray(dq, dtype=float)
        tcp = float(np.linalg.norm(self._jt[:, self.dofs] @ dq)) / self.dt
        elbow = float(np.linalg.norm(self._je[:, self.dofs] @ dq)) / self.dt
        return tcp, elbow

    def scale_step(self, start: np.ndarray, dq: np.ndarray) -> tuple[float, bool, bool]:
        """(scale, tcp_over, elbow_over) for the step `start` -> `start + dq`.

        scale is 1.0 when neither speed is over its limit; otherwise the factor s in (0, 1) that brings the faster
        of the two to its limit, both evaluated with the Jacobians at the commanded target start + dq (the speed
        is linear in dq there). tcp_over / elbow_over say which limit the unscaled step went over."""
        start = np.asarray(start, dtype=float)
        dq = np.asarray(dq, dtype=float)
        tcp, elbow = self.speeds(start + dq, dq)
        tcp_over, elbow_over = tcp > TCP_SPEED_LIMIT, elbow > ELBOW_SPEED_LIMIT
        if not (tcp_over or elbow_over):
            return 1.0, False, False
        return 1.0 / max(tcp / TCP_SPEED_LIMIT, elbow / ELBOW_SPEED_LIMIT), tcp_over, elbow_over


class SafetyMeter:
    """Per-sample TCP contact force, momentum and mechanical power of the arm (see `measure`). The env averages the
    samples of one control step's substeps and keeps the largest control-step mean as the repetition's peak."""

    def __init__(self, model: mujoco.MjModel, part_bodies: dict[str, int]) -> None:
        self.model = model
        gripper = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")
                            .startswith(GRIPPER_PREFIX) for g in range(model.ngeom)])
        self.gripper_geom = gripper
        self.part_of_geom = {int(g): b for b in set(part_bodies.values()) for g in range(model.ngeom)
                             if model.geom_bodyid[g] == b}
        self.robot_root = int(model.body_rootid[model.body("base_link").id])
        self.dofs = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.actuators = np.array([next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == model.joint(j).id)
                                   for j in ARM_JOINTS])
        self._f = np.zeros(6)

    def measure(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(TCP force N, momentum kg m/s, power W) of the current sample.

        - TCP force: magnitude of the vector sum of the contact forces that every non-gripper geom (parts,
          fixture, table, arm links) exerts on the gripper's geoms (bodies under `gripper/`). A grasp's squeeze
          cancels in the sum; what remains is the net external force on the tool, like the UR TCP force.
        - Momentum: |Σ m_i v_i| over the robot (every body of the arm's kinematic tree, gripper included; v_i the
          velocity of body i's centre of mass) plus every part touching a gripper geom (a grasped or pushed part).
        - Power: Σ |τ_i · q̇_i| over the six arm joints, τ_i the joint's actuator force.
        """
        m, d = self.model, data
        force = np.zeros(3)
        touched: set[int] = set()
        ncon = int(d.ncon)
        if ncon:
            geoms = np.asarray(d.contact.geom[:ncon])
            g1, g2 = self.gripper_geom[geoms[:, 0]], self.gripper_geom[geoms[:, 1]]
            for i in np.flatnonzero(g1 != g2):
                mujoco.mj_contactForce(m, d, int(i), self._f)
                frame = np.asarray(d.contact.frame[i]).reshape(3, 3)
                world = frame.T @ self._f[:3]  # on geom2, from geom1
                force += world if g2[i] else -world
                other = int(geoms[i, 0] if g2[i] else geoms[i, 1])
                if other in self.part_of_geom:
                    touched.add(self.part_of_geom[other])
        mujoco.mj_subtreeVel(m, d)
        root = self.robot_root
        momentum = float(m.body_subtreemass[root]) * d.subtree_linvel[root]
        for body in touched:
            momentum = momentum + float(m.body_mass[body]) * d.subtree_linvel[body]
        power = float(np.sum(np.abs(d.actuator_force[self.actuators] * d.qvel[self.dofs])))
        return float(np.linalg.norm(force)), float(np.linalg.norm(momentum)), power
