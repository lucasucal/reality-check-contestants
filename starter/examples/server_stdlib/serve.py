"""Program-form example with the standard library only (http.server). Same contract and the same
example weights as ../server_fastapi/serve.py; use whichever style you like (Flask, aiohttp,
FastAPI, plain sockets ... any Python program that serves docs/api.md).

Copy to the root of your repository and set manifest.json "entry_point": "serve.py".
The program must listen on 0.0.0.0:$RC_POLICY_PORT, keep running, and implement /healthz, /readyz
and /act exactly as docs/api.md says.
"""
from __future__ import annotations

import json
import math
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

CONTRACT = "v0"
SCHEMA_VERSION = 2
CAPABILITIES = {"chunking": True, "caching": False}
MAX_JOINT_DELTA = (0.10472, 0.10472, 0.15708, 0.15708, 0.15708, 0.15708)
STATE: dict[str, Any] = {"model": None, "error": None}


def weights_dir() -> Path:
    return Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")


def read_safetensors(path: Path) -> dict:
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
        out[name] = (info["shape"], list(struct.unpack(f"<{(end - start) // 4}f", raw[base + start:base + end])))
    return out


class ExampleModel:
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

    def act(self, obs: dict) -> dict:
        proprio = obs.get("proprio") if isinstance(obs.get("proprio"), dict) else {}
        raw = proprio.get("joint_pos")
        pos = []
        for item in (raw if isinstance(raw, (list, tuple)) else [])[:6]:
            try:
                x = float(item)
            except (TypeError, ValueError):
                x = 0.0
            pos.append(x if math.isfinite(x) else 0.0)
        pos += [0.0] * (6 - len(pos))
        out = [sum(w * p for w, p in zip(row, pos)) + b for row, b in zip(self.weight, self.bias)]
        values = [max(-m, min(m, v)) for v, m in zip(out[:6], MAX_JOINT_DELTA)]
        return {"type": "joint_delta", "values": values, "gripper": max(0.0, min(1.0, out[6]))}


def load_model() -> None:
    try:
        STATE["model"] = ExampleModel()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        STATE["error"] = "policy failed to load"
        print(f"[example] load FAILED: {type(exc).__name__}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[example] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, status: int, body: dict, headers: dict | None = None) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_json(200, {})
        elif self.path == "/readyz":
            body = {"ready": STATE["model"] is not None, "contract": CONTRACT,
                    "schema_version": SCHEMA_VERSION, "capabilities": dict(CAPABILITIES)}
            if STATE["error"]:
                body["error"] = STATE["error"]
            self.send_json(200, body)
        else:
            self.send_json(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/act":
            return self.send_json(404, {"error": f"no route {self.path}"})
        if STATE["model"] is None:
            return self.send_json(503, {"error": "not ready; weights still loading"})
        try:
            envelope = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        except ValueError as exc:
            return self.send_json(400, {"error": f"invalid JSON body: {exc}"})
        if not isinstance(envelope, dict):
            return self.send_json(400, {"error": "request body must be a JSON object"})
        for field in ("episode_id", "step"):
            if field not in envelope:
                return self.send_json(400, {"error": f"missing required field: {field}"})
        try:
            action = STATE["model"].act(envelope)
        except Exception as exc:
            return self.send_json(500, {"error": f"policy act() failed: {exc}"})
        ep, step = envelope["episode_id"], envelope["step"]
        self.send_json(200, {"episode_id": ep, "step": step, "action": action},
                       {"X-RC-Episode-Id": str(ep), "X-RC-Step": str(step)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("RC_POLICY_PORT") or 8080)), Handler)
    server.daemon_threads = True
    threading.Thread(target=load_model, daemon=True).start()
    server.serve_forever()
