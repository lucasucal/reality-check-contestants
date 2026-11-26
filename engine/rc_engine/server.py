"""Engine HTTP service. Single-threaded HTTPServer: requests are served one at a time, so
time moves only inside /step (lockstep by construction). The socket timeout bounds each
read, so a client that goes silent is dropped after 5 s; one that trickles bytes is not
(acceptable: the runner is the only client, on loopback).

Models compile on a background thread so /healthz answers immediately; the warm-up reset
runs on the serving thread because macOS only allows the first GL context there.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

from .env import Env
from .episode import EngineError, EpisodeManager
from .schema import SCHEMA_VERSION
from .score import SCORER
from .tasks import make_env

MAX_BODY = 1 << 20


def build_info(jpeg_quality: int) -> dict[str, Any]:
    """Renderer facts a replay must match (recorded in the runner's journal header)."""
    info: dict[str, Any] = {"mujoco_gl": os.environ.get("MUJOCO_GL", ""), "jpeg_quality": jpeg_quality,
                            "render_offsamples": os.environ.get("RC_RENDER_OFFSAMPLES") or "0"}  # unset = 0
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except Exception:  # noqa: BLE001 - dummy-only engines may run without MuJoCo
        pass
    try:
        import PIL
        info["pillow"] = PIL.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


class _Handler(BaseHTTPRequestHandler):
    server: "EngineServer"
    timeout = 5.0

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[rc_engine] {self.address_string()} {fmt % args}", flush=True)

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if isinstance(code, int) and code >= 400:
            super().log_request(code, size)

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _require_ready(self) -> EpisodeManager:
        if self.server.manager is None:
            raise EngineError(503, "not_ready", self.server.load_error or "engine is still loading")
        return self.server.manager

    def _read_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise EngineError(400, "bad_request", "Content-Length header required")
        if length < 0:
            raise EngineError(400, "bad_request", "Content-Length must be non-negative")
        if length > MAX_BODY:
            raise EngineError(413, "payload_too_large", f"body exceeds {MAX_BODY} bytes")
        try:
            raw = self.rfile.read(length)
        except TimeoutError:
            raise EngineError(408, "request_timeout", f"body not received within {self.timeout}s")
        if len(raw) < length:
            raise EngineError(400, "bad_request", "body shorter than Content-Length")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise EngineError(400, "bad_request", f"invalid JSON body: {type(exc).__name__}")
        if not isinstance(body, dict):
            raise EngineError(400, "bad_request", "body must be a JSON object")
        return body

    def _route(self) -> dict[str, Any]:
        path = urlsplit(self.path).path
        if self.command == "GET" and path == "/healthz":
            return {}
        if self.command == "GET" and path == "/readyz":
            return self.server.readiness()
        if self.command == "GET" and path == "/tasks":
            manager = self._require_ready()
            quality = self.server.jpeg_quality
            return {"schema_version": SCHEMA_VERSION,
                    "tasks": [env.spec.to_json(jpeg_quality=quality) for env in manager.envs.values()]}
        if self.command == "POST" and path in ("/reset", "/step"):
            manager = self._require_ready()
            body = self._read_body()
            return manager.reset(body) if path == "/reset" else manager.step(body)
        if self.command == "POST" and path == "/render" and self.server.replay_enabled:
            return self._require_ready().render(self._read_body())
        raise EngineError(404, "not_found", f"no route {self.command} {path}")

    def _handle(self) -> None:
        try:
            status, body = 200, self._route()
        except EngineError as err:
            status, body = err.status, {"error": {"code": err.code, "message": str(err)}}
        except Exception as exc:
            traceback.print_exc()
            status, body = 500, {"error": {"code": "engine_error", "message": repr(exc)}}
        try:
            self._send(status, body)
        except OSError:
            self.close_connection = True

    do_GET = do_POST = _handle


class EngineServer(HTTPServer):
    def __init__(self, address: tuple[str, int], task_ids: list[str], jpeg_quality: int = 90,
                 factory: Callable[[str], Env] = make_env) -> None:
        super().__init__(address, _Handler)
        self.task_ids = list(task_ids)
        self.jpeg_quality = jpeg_quality
        # Organizer replay endpoint (POST /render). Off unless RC_ENGINE_REPLAY=1: a judged engine never serves it.
        self.replay_enabled = os.environ.get("RC_ENGINE_REPLAY") == "1"
        self.manager: EpisodeManager | None = None
        self.load_error: str | None = None
        self._loaded: dict[str, Env] | None = None
        threading.Thread(target=self._load, args=(factory,), daemon=True).start()

    def _load(self, factory: Callable[[str], Env]) -> None:
        try:
            self._loaded = {task_id: factory(task_id) for task_id in self.task_ids}
        except Exception as exc:
            self.load_error = f"task load failed: {exc}"

    def service_actions(self) -> None:
        if self.manager is not None or self._loaded is None or self.load_error:
            return
        try:
            for env in self._loaded.values():
                env.reset(0)
        except Exception as exc:
            self.load_error = f"warm-up reset failed: {exc}"
            return
        self.manager = EpisodeManager(self._loaded, self.jpeg_quality)

    def server_close(self) -> None:
        """Stop listening and free the envs' renderers here, on the serving thread that made them: left to
        interpreter exit, EGL teardown prints tracebacks (GB10)."""
        super().server_close()
        for env in (self._loaded or {}).values():
            close = getattr(env, "close", None)
            if close is not None:
                close()

    def readiness(self) -> dict[str, Any]:
        body: dict[str, Any] = {"ready": self.manager is not None, "tasks": self.task_ids, "schema_version": SCHEMA_VERSION,
                                "scorer": SCORER, "build": build_info(self.jpeg_quality)}
        if self.load_error:
            body["error"] = self.load_error
        return body
