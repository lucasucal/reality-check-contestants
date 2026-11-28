"""The policy you edit: a tiny linear policy whose weights are a safetensors file in your Hugging
Face snapshot, mounted read-only at /weights.

Replace load() and act() with your model. The judged image provides torch, safetensors, numpy,
LeRobot and the rest. `manifest.json` names this class as `entry_point` ("policy.example:ExamplePolicy").

Weights (``/weights/model.safetensors``), written by ``scripts/make_example_weights.py``:
    weight  float32 [7, 6]   maps the 6 joint positions to 6 joint deltas and a gripper value
    bias    float32 [7]

Action contract:
    {"type": "joint_delta", "values": [6 floats, |v[i]| <= MAX_JOINT_DELTA], "gripper": float in [0, 1]}
    MAX_JOINT_DELTA: 0.05 rad per joint per control step.

For LeRobot SmolVLA see ``examples/smolvla/smolvla.py``. FP32, BF16 and FP16 weights are the
baseline formats for the judged image (NVIDIA RTX PRO 6000, amd64); see ``runtime/numeric_formats.json`` in
the participant bundle.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rc_sdk.contract import MAX_JOINT_DELTA
from rc_sdk.policy import Policy
WEIGHTS_FILE = "model.safetensors"


def weights_dir() -> Path:
    """The runner exports RC_WEIGHTS_DIR=/weights; load only from there, never from the network."""
    return Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")


class ExamplePolicy(Policy):
    def load(self) -> None:
        from safetensors.torch import load_file  # provided by the image

        path = weights_dir() / WEIGHTS_FILE
        if not path.is_file():
            raise RuntimeError(f"{path} is missing: put {WEIGHTS_FILE} in your Hugging Face model")
        tensors = load_file(str(path))
        self.weight = tensors["weight"].float()
        self.bias = tensors["bias"].float()
        if tuple(self.weight.shape) != (7, 6) or tuple(self.bias.shape) != (7,):
            raise RuntimeError("expected weight [7, 6] and bias [7]")
        print(f"[example] loaded {path}", flush=True)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        import torch

        out = (self.weight @ torch.tensor(obs["proprio"]["joint_pos"], dtype=torch.float32) + self.bias).tolist()
        values = [max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, float(v))) for v in out[:6]]
        gripper = max(0.0, min(1.0, float(out[6])))
        return {"type": "joint_delta", "values": values, "gripper": gripper}
