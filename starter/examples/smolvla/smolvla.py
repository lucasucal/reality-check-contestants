"""Optional example: LeRobot SmolVLA (lerobot/smolvla_base) behind the Policy class.

To use it, copy this file to policy/smolvla.py and set manifest.json "entry_point" to
"policy.smolvla:ExamplePolicy". It loads the snapshot
mounted at /weights. To try it locally without your own weights, set RC_ALLOW_BASE_MODEL_FALLBACK=1
(it then downloads the public base model; never set it in a submission):

    RC_ALLOW_BASE_MODEL_FALLBACK=1 PYTHONPATH=rc_sdk python3 -m rc_sdk \
        --manifest manifest.json --policy policy.smolvla:ExamplePolicy

The adapter maps the camera views to SmolVLA's image slots, passes the instruction and joint
positions, and ignores the three goal views. Its clamping and gripper conversion are specific to this
example.

SmolVLA's own action is not a joint_delta. ``act`` runs ``select_action`` and
translates the vector inside this container. A poor translation is acceptable.
The vector comes from ``select_action`` (SmolVLA may serve later steps from
the action chunk that call already sampled).

Action contract:
    {"type": "joint_delta", "values": [6 floats, |v[i]| <= MAX_JOINT_DELTA], "gripper": float in [0, 1]}

``lerobot/smolvla_base`` emits a length-6 action. The first 6 numbers become
``values`` (clipped). There is no 7th number, so ``gripper`` is 0.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rc_sdk.contract import CAMERA_VIEWS, MAX_JOINT_DELTA
from rc_sdk.policy import Policy

WEIGHTS_DIR = Path("/weights")
MODEL_ID = "lerobot/smolvla_base"
# Camera slots named by the smolvla_base config (input_features).
CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


def _weights_source() -> str:
    """Require mounted checkpoint files unless local fallback is opted into."""
    if WEIGHTS_DIR.is_dir() and any(WEIGHTS_DIR.iterdir()):
        return str(WEIGHTS_DIR)
    if os.environ.get("RC_ALLOW_BASE_MODEL_FALLBACK") == "1":
        print("[example] WARNING: using development-only public base model fallback", flush=True)
        return MODEL_ID
    raise RuntimeError("pinned weights are missing from /weights")


def _jpeg_chw(payload: str, view: str) -> torch.Tensor:
    """Decode one live camera JPEG into a float CHW tensor in [0, 1]."""
    with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as image:
        if image.format != "JPEG" or image.size != (640, 360):
            raise ValueError(f"images.{view} must be a 640x360 JPEG")
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def _observation_images(obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map the named live views to the checkpoint's camera slots."""
    return {key: _jpeg_chw(obs["images"][view], view) for key, view in zip(CAMERA_KEYS, CAMERA_VIEWS)}


def _first_action_row(action: Any) -> torch.Tensor:
    """Flatten select_action's first action row to a 1-D float tensor."""
    if isinstance(action, dict):
        action = action["action"]
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32)
    vector = action.detach().to(dtype=torch.float32, device="cpu")
    while vector.ndim > 1:
        vector = vector[0]
    return vector.reshape(-1)


def _joint_delta(vector: torch.Tensor) -> dict[str, Any]:
    """First 6 numbers -> joint deltas within MAX_JOINT_DELTA (per joint). The next number -> gripper.

    The base model returns six values. That adapter uses a closed gripper target.
    """
    if vector.numel() not in (6, 7) or not torch.isfinite(vector).all():
        raise ValueError("SmolVLA checkpoint must return six or seven finite values")
    clipped = [max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, v)) for v in vector[:6].tolist()]
    gripper = float(vector[6].clamp(0.0, 1.0).item()) if vector.numel() == 7 else 0.0
    return {
        "type": "joint_delta",
        "values": clipped,
        "gripper": gripper,
    }


class ExamplePolicy(Policy):
    """SmolVLA behind the reality-check container contract."""

    def load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        source = _weights_source()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[example] loading SmolVLA from {source} on {device}", flush=True)
        config = PreTrainedConfig.from_pretrained(source)
        config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(source, config=config)
        self.policy.eval()
        # Saved preprocessor tokenizes `task` into observation.language.tokens,
        # which select_action reads. Override the checkpoint's cuda device.
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=source,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.policy.reset()
        self._episode: Any = None
        self._memo: tuple[Any, Any, dict[str, Any]] | None = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        episode_id = obs.get("episode_id")
        step = obs.get("step")
        if self._memo is not None and self._memo[0] == episode_id and self._memo[1] == step:
            return self._memo[2]
        if episode_id != self._episode or step == 0:
            self._episode = episode_id
            self.policy.reset()

        # `goal` contains three pictures of the finished scene. SmolVLA's cameras are live
        # views, so the goal is not fed in. The instruction is passed under
        # SmolVLA's `task` key.
        frame: dict[str, Any] = {
            "observation.state": torch.tensor(obs["proprio"]["joint_pos"], dtype=torch.float32),
            "task": obs["instruction"],
        }
        frame.update(_observation_images(obs))
        batch = self.preprocess(frame)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        out = _joint_delta(_first_action_row(action))
        self._memo = (episode_id, step, out)
        return out
