"""PolicyServer — HTTP shim implementing contract v0 (docs/platform.md §6.1).

Stdlib only (http.server / json). Run with: python -m rc_sdk
[--manifest manifest.json] [--policy your.module:YourPolicy]
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional

from .contract import CAPABILITIES, CONTRACT, SCHEMA_VERSION, check_manifest_schema, observation_error
from .policy import Policy
from .transport import PolicyHTTPServer, tls_server_context

DEFAULT_PORT = 8080  # the port the judged image serves on


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_policy(spec: str) -> Policy:
    """'package.module:ClassName' -> instance."""
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"--policy must be 'module:Class', got {spec!r}")
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)
    return cls()


class _Handler(BaseHTTPRequestHandler):
    server: "PolicyServer"

    def log_message(self, fmt: str, *args: Any) -> None:  # runner captures stdout
        print(f"[rc_sdk] {self.address_string()} {fmt % args}", flush=True)

    # -- helpers ---------------------------------------------------------
    def _send_json(self, status: int, body: Dict[str, Any],
                   extra_headers: Optional[Dict[str, str]] = None) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/healthz":
            # LIVENESS ONLY: must respond within 5 s of process start,
            # BEFORE weights load. Never gate on readiness here.
            self._send_json(200, {})
        elif self.path == "/readyz":
            body = {
                "ready": self.server.ready.is_set(),
                "contract": CONTRACT,
                "schema_version": SCHEMA_VERSION,
                "capabilities": dict(CAPABILITIES),
            }
            if self.server.load_error is not None:
                body["error"] = "policy failed to load"
            self._send_json(200, body)
        else:
            self._send_json(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send_json(404, {"error": f"no route {self.path}"})
            return
        if not self.server.ready.is_set():
            self._send_json(503, {"error": "not ready; weights still loading"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must be nonnegative")
            envelope = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        error = observation_error(envelope)
        if error:
            self._send_json(400, {"error": error})
            return
        episode_id, step = envelope["episode_id"], envelope["step"]
        try:
            action = self.server.policy.act(envelope)
        except Exception as exc:  # surface a readable failure, keep server up
            self._send_json(500, {"error": f"policy act() failed: {exc}"})
            return
        try:
            json.dumps(action)
        except (TypeError, ValueError) as exc:
            print(f"[rc_sdk] /act step {step}: action is not JSON-serializable: {exc}", flush=True)
            self._send_json(500, {"error": f"action is not JSON-serializable: {exc}"})
            return
        # CONTRACT: response echoes episode_id and step VERBATIM; headers
        # X-RC-Episode-Id / X-RC-Step ride along for dedupe/logging.
        self._send_json(200, {"episode_id": episode_id, "step": step, "action": action},
                        {"X-RC-Episode-Id": str(episode_id), "X-RC-Step": str(step)})


class PolicyServer(PolicyHTTPServer):
    """HTTPServer + background load thread; ready flips after load() returns."""

    daemon_threads = True

    def __init__(self, port: int, policy: Policy, *, tls_context=None) -> None:
        super().__init__(("0.0.0.0", port), _Handler, tls_context=tls_context)
        self.policy = policy
        self.ready = threading.Event()
        self.load_error: Optional[BaseException] = None
        self._load_thread = threading.Thread(target=self._load, daemon=True)

    def _load(self) -> None:
        try:
            self.policy.load()
        except Exception as exc:
            self.load_error = exc
            import traceback
            print("[rc_sdk] load() FAILED; full diagnostic follows:", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return
        self.ready.set()
        print("[rc_sdk] ready", flush=True)

    def start_load(self) -> None:
        self._load_thread.start()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="rc_sdk")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--policy",
                        help="'module:Class' of your Policy subclass (default: manifest entry_point)")
    parser.add_argument("--port", type=int, default=None,
                        help=f"default: manifest.json port, else {DEFAULT_PORT}")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        check_manifest_schema(manifest)
        entry = args.policy or manifest.get("entry_point")
        if not isinstance(entry, str):
            raise ValueError("set manifest entry_point or --policy to 'module:Class'")
        port = args.port if args.port is not None else int(manifest.get("port", DEFAULT_PORT))
        tls_context = tls_server_context(os.environ)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    policy = import_policy(entry)
    server = PolicyServer(port, policy, tls_context=tls_context)
    server.start_load()  # healthz answers immediately; readyz flips after load
    transport = "HTTPS" if tls_context else "HTTP"
    print(f"[rc_sdk] serving {transport} contract {CONTRACT}, schema {SCHEMA_VERSION}, on port {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
