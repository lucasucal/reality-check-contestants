"""dummy_v0 (M1): scripted env with no physics; proves the service contract without MuJoCo."""
from __future__ import annotations

import numpy as np

from ..env import StepResult, hash_state
from ..schema import NUM_JOINTS, CameraSpec, JointDelta, RawObs, TaskSpec

SPEC = TaskSpec(
    task_id="dummy_v0",
    description="Scripted stand-in: drive the 6 joints to a seeded target. No physics.",
    episode_length=50,
    control_hz=20,
    cameras=(CameraSpec("front", 64, 48),),
)
SUCCESS_TOL = 0.01


class DummyEnv:
    spec = SPEC

    def __init__(self) -> None:
        self.step_count = 0
        self.joint_pos = np.zeros(NUM_JOINTS)
        self.joint_vel = np.zeros(NUM_JOINTS)
        self.target = np.zeros(NUM_JOINTS)
        self.gripper = 1.0

    def reset(self, seed: int) -> RawObs:
        rng = np.random.Generator(np.random.PCG64(seed))
        self.joint_pos = rng.uniform(-1.0, 1.0, NUM_JOINTS)
        self.target = rng.uniform(-1.0, 1.0, NUM_JOINTS)
        self.joint_vel = np.zeros(NUM_JOINTS)
        self.gripper = 1.0
        self.step_count = 0
        return self._obs()

    def step(self, action: JointDelta) -> StepResult:
        delta = np.array(action.values)
        self.joint_pos = self.joint_pos + delta
        self.joint_vel = delta * SPEC.control_hz
        self.gripper = action.gripper
        self.step_count += 1
        success = bool(np.max(np.abs(self.joint_pos - self.target)) < SUCCESS_TOL)
        return StepResult(self._obs(), success or self.step_count >= SPEC.episode_length, success)

    def state_hash(self) -> str:
        return hash_state(self.step_count, self.joint_pos, self.joint_vel, self.target, np.array([self.gripper]))

    def _obs(self) -> RawObs:
        cam = SPEC.cameras[0]
        colour = (np.mod(self.joint_pos[:3], 1.0) * 255).astype(np.uint8)
        frame = np.broadcast_to(colour, (cam.height, cam.width, 3)).copy()
        return RawObs(
            images={cam.name: frame},
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
            gripper=self.gripper,
            ee_pos=self.joint_pos[:3].copy(),
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
