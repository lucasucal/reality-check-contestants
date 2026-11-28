"""Program-form example: your submission is a black box, any Python program that serves the
policy HTTP contract (docs/api.md). This one uses FastAPI + uvicorn (both are in the image).

How to use it: copy this file to the root of your repository, set manifest.json to
    "entry_point": "serve.py"
and replace `ExampleModel` with your own model code. rc_runtime sets up the sandbox (cwd /submission,
TMPDIR, sys.path) and then runs this file as __main__ in the same interpreter. The program must
    * listen on 0.0.0.0:$RC_POLICY_PORT (always 8080 in the judged image),
    * answer GET /healthz immediately, GET /readyz (ready only after the weights are loaded),
      and POST /act exactly as docs/api.md describes (echo episode_id and step verbatim in the
      body and in the X-RC-Episode-Id / X-RC-Step headers),
    * keep running: if it returns or raises, the run fails ("entry program exited").
Only Python and the image's packages run. No network; /submission and /weights are read-only;
write under /scratch.

Weights: /weights/model.safetensors (scripts/make_example_weights.py): float32 weight [7, 6] and
bias [7]. Read here with the standard library so the example needs nothing but FastAPI; a real
policy would use safetensors.torch.load_file and torch.
"""
from __future__ import annotations

import json
import os
import struct
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rc_sdk.contract import CAPABILITIES, CONTRACT, MAX_JOINT_DELTA, SCHEMA_VERSION, observation_error


def weights_dir() -> Path:
    return Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")


def read_safetensors(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    """Minimal float32 safetensors reader: 8-byte header size, JSON header, raw little-endian data."""
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + header_len])
    base = 8 + header_len
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise RuntimeError(f"{name}: expected F32, got {info['dtype']}")
        start, end = info["data_offsets"]
        n = (end - start) // 4
        out[name] = (info["shape"], list(struct.unpack(f"<{n}f", raw[base + start:base + end])))
    return out


class ExampleModel:
    """Linear policy: 6 joint positions -> 6 joint deltas + gripper."""

    def __init__(self) -> None:
        path = weights_dir() / "model.safetensors"
        if not path.is_file():
            raise RuntimeError(f"{path} is missing: put model.safetensors in your Hugging Face model")
        tensors = read_safetensors(path)
        (w_shape, w), (b_shape, b) = tensors["weight"], tensors["bias"]
        if w_shape != [7, 6] or b_shape != [7]:
            raise RuntimeError("expected weight [7, 6] and bias [7]")
        self.weight = [w[i * 6:(i + 1) * 6] for i in range(7)]
        self.bias = b
        print(f"[example] loaded {path}", flush=True)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        pos = obs["proprio"]["joint_pos"]
        out = [sum(w * p for w, p in zip(row, pos)) + b for row, b in zip(self.weight, self.bias)]
        values = [max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, v)) for v in out[:6]]
        return {"type": "joint_delta", "values": values, "gripper": max(0.0, min(1.0, out[6]))}


app = FastAPI()
state: dict[str, Any] = {"model": None, "error": None}


def load_model() -> None:
    try:
        state["model"] = ExampleModel()
    except Exception as exc:  # keep the server up; /readyz reports a bounded message
        import traceback
        traceback.print_exc()
        state["error"] = "policy failed to load"
        print(f"[example] load FAILED: {type(exc).__name__}", flush=True)


@app.get("/healthz")
def healthz() -> dict:
    return {}  # liveness only: never wait for the model


@app.get("/readyz")
def readyz() -> dict:
    body: dict[str, Any] = {"ready": state["model"] is not None, "contract": CONTRACT,
                            "schema_version": SCHEMA_VERSION, "capabilities": dict(CAPABILITIES)}
    if state["error"]:
        body["error"] = state["error"]
    return body


@app.post("/act")
async def act(request: Request) -> JSONResponse:
    if state["model"] is None:
        return JSONResponse({"error": "not ready; weights still loading"}, status_code=503)
    try:
        envelope = json.loads(await request.body() or b"{}")
    except ValueError as exc:
        return JSONResponse({"error": f"invalid JSON body: {exc}"}, status_code=400)
    error = observation_error(envelope)
    if error:
        return JSONResponse({"error": error}, status_code=400)
    try:
        action = state["model"].act(envelope)
    except Exception as exc:
        return JSONResponse({"error": f"policy act() failed: {exc}"}, status_code=500)
    episode_id, step = envelope["episode_id"], envelope["step"]
    return JSONResponse({"episode_id": episode_id, "step": step, "action": action},
                        headers={"X-RC-Episode-Id": str(episode_id), "X-RC-Step": str(step)})


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    port = int(os.environ.get("RC_POLICY_PORT") or 8080)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
