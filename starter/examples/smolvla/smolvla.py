"""Optional example: LeRobot SmolVLA (lerobot/smolvla_base) behind the Policy class.

To use it, copy this file to policy/smolvla.py and set manifest.json "entry_point" to
"policy.smolvla:ExamplePolicy". It loads the snapshot
mounted at /weights. To try it locally without your own weights, set RC_ALLOW_BASE_MODEL_FALLBACK=1
(it then downloads the public base model; never set it in a submission):

    RC_ALLOW_BASE_MODEL_FALLBACK=1 PYTHONPATH=rc_sdk python3 -m rc_sdk \
        --manifest manifest.json --policy policy.smolvla:ExamplePolicy

The adapter maps the camera views to SmolVLA's image slots, passes the instruction and joint
positions, and ignores the goal image. Its clamping and gripper conversion are specific to this
example.

SmolVLA's own action is not a joint_delta. ``act`` runs ``select_action`` and
translates the vector inside this container. A poor translation is acceptable.
The vector comes from ``select_action`` (SmolVLA may serve later steps from
the action chunk that call already sampled).

Action contract:
    {"type": "joint_delta", "values": [6 floats, |v[i]| <= MAX_JOINT_DELTA[i]], "gripper": float in [0, 1]}

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

from rc_sdk.policy import Policy

WEIGHTS_DIR = Path("/weights")
MODEL_ID = "lerobot/smolvla_base"
# Camera slots named by the smolvla_base config (input_features).
CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
# Match the public camera order.
IMAGE_KEY_ORDER = ("wrist", "left", "right")
# Decoded frame size (height, width). The wrist, left and right Gemini 336 views are
# 640x360 (16:9).
FRAME_HW = {"wrist": (360, 640), "left": (360, 640), "right": (360, 640)}
STATE_DIM = 6
MAX_JOINT_DELTA = (0.10472, 0.10472, 0.15708, 0.15708, 0.15708, 0.15708)
# TokenizerProcessorStep rejects an empty task. Used only when instruction is absent.
_FALLBACK_INSTRUCTION = "move the arm"


def _weights_source() -> str:
    """Require mounted checkpoint files unless local fallback is opted into."""
    if WEIGHTS_DIR.is_dir() and any(WEIGHTS_DIR.iterdir()):
        return str(WEIGHTS_DIR)
    if os.environ.get("RC_ALLOW_BASE_MODEL_FALLBACK") == "1":
        print("[example] WARNING: using development-only public base model fallback", flush=True)
        return MODEL_ID
    raise RuntimeError("pinned weights are missing from /weights")


def _jpeg_chw(payload: str, camera: str = "wrist") -> torch.Tensor:
    """Base64 JPEG -> float CHW in [0, 1], no batch dimension.

    A frame that will not decode becomes a black image of that camera's size so
    SmolVLA still runs.
    """
    try:
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    except Exception:
        return torch.zeros(3, *FRAME_HW[camera], dtype=torch.float32)


def _image_payloads(obs: dict[str, Any]) -> list[tuple[str, str]]:
    """(camera, base64 JPEG) in IMAGE_KEY_ORDER for the cameras present."""
    images = obs.get("images") or {}
    if not isinstance(images, dict):
        return []
    payloads: list[tuple[str, str]] = []
    for key in IMAGE_KEY_ORDER:
        value = images.get(key)
        if isinstance(value, str) and value:
            payloads.append((key, value))
    return payloads


def _observation_images(obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map policy frames onto SmolVLA's camera slots. At least camera1 is present."""
    payloads = _image_payloads(obs)
    tensors: dict[str, torch.Tensor] = {}
    for key, (camera, payload) in zip(CAMERA_KEYS, payloads):
        tensors[key] = _jpeg_chw(payload, camera)
    if CAMERA_KEYS[0] not in tensors:
        tensors[CAMERA_KEYS[0]] = torch.zeros(3, *FRAME_HW["wrist"], dtype=torch.float32)
    return tensors


def _state(obs: dict[str, Any]) -> torch.Tensor:
    """6-D state the checkpoint normalizer expects. UR12e joint_pos, padded."""
    proprio = obs.get("proprio") if isinstance(obs.get("proprio"), dict) else {}
    joint_pos = proprio.get("joint_pos") if isinstance(proprio, dict) else None
    values: list[float] = []
    if isinstance(joint_pos, (list, tuple)):
        for item in list(joint_pos)[:STATE_DIM]:
            try:
                number = float(item)
            except (TypeError, ValueError):
                number = 0.0
            if number != number or number in (float("inf"), float("-inf")):
                number = 0.0
            values.append(number)
    if len(values) < STATE_DIM:
        values.extend([0.0] * (STATE_DIM - len(values)))
    return torch.tensor(values, dtype=torch.float32)


def _instruction(obs: dict[str, Any]) -> str:
    text = obs.get("instruction")
    if isinstance(text, str) and text.strip():
        return text
    return _FALLBACK_INSTRUCTION


def _first_action_row(action: Any) -> torch.Tensor:
    """Flatten select_action's first action row to a 1-D float tensor."""
    if isinstance(action, dict):
        action = action["action"]
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32)
    vector = action.detach().to(dtype=torch.float32, device="cpu")
    while vector.ndim > 1:
        vector = vector[0]
    return torch.nan_to_num(vector.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def _joint_delta(vector: torch.Tensor) -> dict[str, Any]:
    """First 6 numbers -> joint deltas within MAX_JOINT_DELTA (per joint). The next number -> gripper.

    A shorter vector is padded with zeros. No 7th number means gripper 0.
    """
    if vector.numel() < 6:
        vector = torch.nn.functional.pad(vector, (0, 6 - vector.numel()))
    # Clip in Python floats against the float64 per-joint bounds the engine checks.
    clipped = []
    for value, limit in zip(vector[:6].tolist(), MAX_JOINT_DELTA):
        number = float(value)
        if number > limit:
            number = limit
        elif number < -limit:
            number = -limit
        clipped.append(number)
    gripper = float(vector[6].clamp(0.0, 1.0).item()) if vector.numel() > 6 else 0.0
    if gripper > 1.0:
        gripper = 1.0
    elif gripper < 0.0:
        gripper = 0.0
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

        # `goal` is a picture of the finished scene. SmolVLA's cameras are live
        # views, so the goal is not fed in. The instruction is passed under
        # SmolVLA's `task` key.
        frame: dict[str, Any] = {
            "observation.state": _state(obs),
            "task": _instruction(obs),
        }
        frame.update(_observation_images(obs))
        batch = self.preprocess(frame)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        out = _joint_delta(_first_action_row(action))
        self._memo = (episode_id, step, out)
        return out
