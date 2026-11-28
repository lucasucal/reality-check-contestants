"""dummy_v0 (M1): scripted env with no physics; proves the service contract without MuJoCo."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from ..env import StepResult, hash_state
from ..schema import NUM_JOINTS, POLICY_CAMERAS, GoalSet, JointDelta, RawObs, TaskSpec

SPEC = TaskSpec(
    task_id="dummy_v0",
    description="Scripted stand-in: drive the 6 joints to a seeded target. No physics.",
    episode_length=50,
    control_hz=20,
    cameras=POLICY_CAMERAS,
)
SUCCESS_TOL = 0.01


class DummyEnv:
    def __init__(self, spec: TaskSpec = SPEC) -> None:
        self.spec = replace(spec, nominal_goal=GoalSet(), episode_goal=GoalSet())
        target = np.random.Generator(np.random.PCG64(0)).uniform(-1.0, 1.0, (2, NUM_JOINTS))[1]
        self._stage(self.spec.nominal_goal, target, 0)
        self.step_count = 0
        self.joint_pos = np.zeros(NUM_JOINTS)
        self.joint_vel = np.zeros(NUM_JOINTS)
        self.target = np.zeros(NUM_JOINTS)
        self.gripper = 1.0

    def reset(self, seed: int) -> RawObs:
        rng = np.random.Generator(np.random.PCG64(seed))
        self.joint_pos = rng.uniform(-1.0, 1.0, NUM_JOINTS)
        self.target = rng.uniform(-1.0, 1.0, NUM_JOINTS)
        self._stage(self.spec.episode_goal, self.target, seed)
        self.joint_vel = np.zeros(NUM_JOINTS)
        self.gripper = 1.0
        self.step_count = 0
        return self._obs()

    def step(self, action: JointDelta) -> StepResult:
        delta = np.array(action.values)
        self.joint_pos = self.joint_pos + delta
        self.joint_vel = delta * self.spec.control_hz
        self.gripper = action.gripper
        self.step_count += 1
        success = bool(np.max(np.abs(self.joint_pos - self.target)) < SUCCESS_TOL)
        return StepResult(self._obs(), success or self.step_count >= self.spec.episode_length, success)

    def state_hash(self) -> str:
        return hash_state(self.step_count, self.joint_pos, self.joint_vel, self.target, np.array([self.gripper]))

    @staticmethod
    def _frames(joints: np.ndarray) -> dict[str, np.ndarray]:
        """Three projections of the same scripted joint state in this synthetic API fixture."""
        angles = np.cumsum(joints)
        points = np.cumsum(np.column_stack([np.cos(angles), np.sin(angles), np.sin(joints)]), axis=0)
        points = np.vstack([np.zeros(3), points])
        frames = {}
        for camera, (u, v) in zip(POLICY_CAMERAS, ((0, 1), (1, 2), (0, 2))):
            image = Image.new("RGB", (camera.width, camera.height), (38, 48, 58))
            draw = ImageDraw.Draw(image)
            xy = [(camera.width / 2 + 25 * p[u], camera.height / 2 - 25 * p[v]) for p in points]
            draw.line(xy, fill=(220, 180, 110), width=5)
            frames[camera.name] = np.asarray(image)
        return frames

    def _stage(self, goal: GoalSet, target: np.ndarray, seed: int) -> None:
        goal.stage(lambda: self._frames(target), {"seed": seed, "variant": 0,
                   "wrist_joint_pos": target.tolist()}, eager=False)

    def _obs(self) -> RawObs:
        return RawObs(
            images=self._frames(self.joint_pos),
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
            gripper=self.gripper,
            ee_pos=self.joint_pos[:3].copy(),
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
