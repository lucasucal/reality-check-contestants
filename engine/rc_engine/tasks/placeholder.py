"""Placeholder tasks. The pipe runs before a real scene exists.

Same stand-in as dummy_v0: no physics, no objects, no manipulation scene.
The three projections show the current joints and the scripted target joints.
"""
from __future__ import annotations

from ..schema import POLICY_CAMERAS, TaskSpec
from .dummy import DummyEnv

EPISODE_LENGTH = 6
CONTROL_HZ = 20
PUBLIC_IDS = ("placeholder_public_1", "placeholder_public_2", "placeholder_public_3")
HIDDEN_IDS = ("placeholder_hidden_1", "placeholder_hidden_2", "placeholder_hidden_3")
WIRE_IDS = ("1", "2", "3")


def public_spec(index: int) -> TaskSpec:
    return TaskSpec(
        task_id=f"placeholder_public_{index}",
        description="Public placeholder. No scene has been set.",
        episode_length=EPISODE_LENGTH,
        control_hz=CONTROL_HZ,
        cameras=POLICY_CAMERAS,
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
        cameras=POLICY_CAMERAS,
        family="placeholder",
        visibility=visibility,
        instruction=instruction or f"Placeholder hidden task {wire_id}. No scene has been set.",
        wire_id=wire_id,
    )


PUBLIC = tuple(public_spec(i) for i in (1, 2, 3))


class PlaceholderEnv(DummyEnv):
    """Scripted joint-target task with public or organizer-supplied instructions."""
