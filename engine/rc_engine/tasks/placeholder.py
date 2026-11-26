"""Placeholder tasks. The pipe runs before a real scene exists.

Same stand-in as dummy_v0: no physics, no objects, no manipulation scene.
The live camera is a flat frame that follows the joints. The goal is a separate
640×480 JPEG, identical on every step.
"""
from __future__ import annotations

import numpy as np

from ..env import StepResult, hash_state
from ..schema import NUM_JOINTS, CameraSpec, JointDelta, RawObs, TaskSpec

EPISODE_LENGTH = 6
CONTROL_HZ = 20
# Same policy-camera slot the contract already uses: 640×480, key "top".
# The pixels are the stand-in joint colour, not a scene.
CAMERA = CameraSpec("top", 640, 480)
SUCCESS_TOL = 0.01

PUBLIC_IDS = ("placeholder_public_1", "placeholder_public_2", "placeholder_public_3")
HIDDEN_IDS = ("placeholder_hidden_1", "placeholder_hidden_2", "placeholder_hidden_3")
WIRE_IDS = ("1", "2", "3")


def public_spec(index: int) -> TaskSpec:
    return TaskSpec(
        task_id=f"placeholder_public_{index}",
        description="Public placeholder. No scene has been set.",
        episode_length=EPISODE_LENGTH,
        control_hz=CONTROL_HZ,
        cameras=(CAMERA,),
        family="placeholder",
        visibility="public",
        instruction=f"Placeholder public task {index}. No scene has been set.",
    )


def hidden_spec(task_id: str, *, instruction: str, description: str, wire_id: str,
                visibility: str = "hidden") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        description=description or "Hidden placeholder. No scene has been set.",
        episode_length=EPISODE_LENGTH,
        control_hz=CONTROL_HZ,
        cameras=(CAMERA,),
        family="placeholder",
        visibility=visibility,
        instruction=instruction or f"Placeholder hidden task {wire_id}. No scene has been set.",
        wire_id=wire_id,
    )


PUBLIC = tuple(public_spec(i) for i in (1, 2, 3))


class PlaceholderEnv:
    """Scripted joints, no physics. Success is the dummy joint-target test."""

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
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
        self.joint_vel = delta * self.spec.control_hz
        self.gripper = action.gripper
        self.step_count += 1
        success = bool(np.max(np.abs(self.joint_pos - self.target)) < SUCCESS_TOL)
        done = success or self.step_count >= self.spec.episode_length
        return StepResult(self._obs(), done, success)

    def state_hash(self) -> str:
        return hash_state(self.step_count, self.joint_pos, self.joint_vel, self.target, np.array([self.gripper]))

    def _obs(self) -> RawObs:
        cam = self.spec.cameras[0]
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
