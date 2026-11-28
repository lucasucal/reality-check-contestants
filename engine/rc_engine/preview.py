"""In-process watch loop for the dashboard.

The scene frames come from the engine's policy and audience cameras. The goal
image sits beside it and is the same JPEG that /act carries. The packet is that
/act body: no success bit. Organizer modes require a mounted task pack.

    python -m rc_engine.preview [--port 8091] [--task-packs /packs]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .env import Env
from .packs import PackError, load_packs
from .schema import JointDelta, encode_obs, encode_views, parse_action
from .tasks import Catalog, builtin_catalog
from .tasks.tabletop import PUBLIC_IDS

EPISODES = 4
SEEDS = (0, 1, 2, 3)


def task_mean(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return round(sum(1 for flag in flags if flag) / len(flags), 6)


def contest_mean(groups: list[list[bool]]) -> float:
    """Mean of the per-task means. One number. No per-task report."""
    means = [task_mean(group) for group in groups if group]
    if not means:
        return 0.0
    return round(sum(means) / len(means), 6)


def _make_envs(catalog: Catalog, task_ids: tuple[str, ...]) -> dict[str, Env]:
    # Side cameras are for this watch. Scoring and training leave the flag unset.
    os.environ["RC_AUDIENCE_CAMERAS"] = "1"
    return {task_id: catalog.make(task_id) for task_id in task_ids}


def _order(envs: dict[str, Env], mode: str) -> list[str]:
    public = [task_id for task_id, env in envs.items() if env.spec.visibility != "hidden"]
    hidden = [task_id for task_id, env in envs.items() if env.spec.visibility == "hidden"]
    if mode == "public":
        return public[:1]
    if mode == "hidden":
        return hidden[:1]
    return public + hidden


def _act(step: int) -> dict[str, Any]:
    """A legal joint_delta. It does not solve the stand-in. The scene still moves."""
    sign = 1.0 if step % 2 == 0 else -1.0
    return {"type": "joint_delta", "values": [sign * 0.02] * 6, "gripper": 1.0 if step % 2 == 0 else 0.0}


def _packet(obs: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """The /act body. success is not a field."""
    packet = {"episode_id": episode_id, **obs}
    packet.pop("success", None)
    return packet


def _views(env: Env) -> dict[str, str] | None:
    """45° check cameras. They are not part of the /act packet."""
    frames = getattr(env, "audience_images", None) or {}
    if not frames:
        return None
    return encode_views(frames, 80)


def _scene(obs: dict[str, Any]) -> str | None:
    images = obs.get("images") or {}
    if not images:
        return None
    return next(iter(images.values()))


def _blank(mode: str | None = None) -> dict[str, Any]:
    return {
        "status": "idle",
        "mode": mode,
        "scene": None,
        "views": None,
        "goal": None,
        "packet": None,
        "action": None,
        "task_id": None,
        "episode": None,
        "episodes_per_task": EPISODES,
        "step": None,
        "episode_length": None,
        "running_average": None,
        "contest_average": None,
    }


def _aim_env(env: Env, aim: Callable[[], tuple[float, float]] | None) -> None:
    if aim is None or not hasattr(env, "set_orbit"):
        return
    azimuth, elevation = aim()
    env.set_orbit(azimuth, elevation)  # type: ignore[attr-defined]


def run_view(task_id: str, publish: Callable[[dict[str, Any]], None], *,
             stop: threading.Event, aim: Callable[[], tuple[float, float]],
             catalog: Catalog | None = None) -> None:
    """Hold one scene on seed 0 and re-render while the orbit camera moves."""
    env = _make_envs(catalog or builtin_catalog(), (task_id,))[task_id]
    camera = aim()
    _aim_env(env, lambda: camera)
    raw = env.reset(0)
    obs = encode_obs(env.spec, 0, raw, 90)
    episode_id = uuid.uuid4().hex
    while not stop.is_set():
        packet = _packet(obs, episode_id)
        publish({
            "status": "running",
            "mode": "view",
            "scene": _scene(obs),
            "views": _views(env),
            "goal": obs["goal"]["wrist"],
            "packet": packet,
            "action": None,
            "task_id": packet.get("task_id"),
            "scene_id": task_id,
            "episode": 1,
            "episodes_per_task": 1,
            "step": 0,
            "episode_length": env.spec.episode_length,
            "running_average": None,
            "contest_average": None,
        })
        while not stop.wait(0.12):
            next_camera = aim()
            if next_camera != camera:
                camera = next_camera
                _aim_env(env, lambda: camera)
                obs = encode_obs(env.spec, 0, env._obs(), 90)  # noqa: SLF001 - re-render only
                break


def run_loop(mode: str, publish: Callable[[dict[str, Any]], None], *, pace: float = 0.0,
             stop: threading.Event | None = None,
             envs: dict[str, Env] | None = None,
             catalog: Catalog | None = None,
             aim: Callable[[], tuple[float, float]] | None = None) -> None:
    """Run four episodes per selected task.

    Public and hidden mode select the first task in their group. Contest mode
    selects both groups and includes contest_average in the last frame.
    """
    if mode not in {"public", "hidden", "contest"}:
        raise ValueError(f"unknown mode {mode!r}")
    stop = stop or threading.Event()
    if envs is None:
        catalog = catalog or builtin_catalog()
        pack_ids = tuple(task_id for task_id in catalog.defaults if task_id not in PUBLIC_IDS)
        if mode != "public" and not pack_ids:
            raise ValueError("organizer mode requires a task pack")
        if mode == "public":
            wanted = PUBLIC_IDS[:1]
        elif mode == "hidden":
            wanted = pack_ids[:1]
        else:
            wanted = PUBLIC_IDS + pack_ids
        envs = _make_envs(catalog, wanted)
    order = _order(envs, mode)
    groups: list[list[bool]] = []
    last: dict[str, Any] = _blank(mode)
    for task_id in order:
        if stop.is_set():
            return
        env = envs[task_id]
        successes: list[bool] = []
        for index, seed in enumerate(SEEDS, start=1):
            if stop.is_set():
                return
            last = _run_episode(env, seed, index, successes, publish, pace, stop, mode, last, aim)
        groups.append(successes)
        if successes:
            last = {**last, "running_average": task_mean(successes), "episode": EPISODES, "status": "running"}
            publish(last)
    last = {
        **last,
        "status": "done",
        "running_average": task_mean(groups[-1]) if groups else None,
        "contest_average": contest_mean(groups) if mode == "contest" else None,
    }
    publish(last)


def _run_episode(env: Env, seed: int, index: int, successes: list[bool],
                 publish: Callable[[dict[str, Any]], None], pace: float, stop: threading.Event,
                 mode: str, previous: dict[str, Any],
                 aim: Callable[[], tuple[float, float]] | None = None) -> dict[str, Any]:
    _aim_env(env, aim)
    raw = env.reset(seed)
    obs = encode_obs(env.spec, 0, raw, 90)
    episode_id = uuid.uuid4().hex
    step = 0
    action: dict[str, Any] | None = None
    frame = previous
    while step < env.spec.episode_length:
        if stop.is_set():
            return frame
        action = _act(step)
        packet = _packet(obs, episode_id)
        frame = {
            "status": "running",
            "mode": mode,
            "scene": _scene(obs),
            "views": _views(env),
            "goal": obs["goal"]["wrist"],
            "packet": packet,
            "action": action,
            "task_id": packet.get("task_id"),
            "episode": index,
            "episodes_per_task": EPISODES,
            "step": step,
            "episode_length": env.spec.episode_length,
            "running_average": task_mean(successes) if successes else None,
            "contest_average": None,
        }
        publish(frame)
        if pace > 0:
            time.sleep(pace)
        _aim_env(env, aim)
        parsed: JointDelta = parse_action(action)[0]
        result = env.step(parsed)
        step = env.step_count
        obs = encode_obs(env.spec, step, result.obs, 90)
        if result.done:
            successes.append(bool(result.success))
            frame = {**frame, "running_average": task_mean(successes)}
            publish(frame)
            return frame
    successes.append(False)
    frame = {**frame, "running_average": task_mean(successes)}
    publish(frame)
    return frame


class PreviewState:
    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog or builtin_catalog()
        self._lock = threading.Lock()
        self._frame = _blank()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._azimuth = 210.0
        self._elevation = 32.0

    def aim_now(self) -> tuple[float, float]:
        with self._lock:
            return self._azimuth, self._elevation

    def aim(self, azimuth: float, elevation: float) -> None:
        with self._lock:
            self._azimuth = float(azimuth) % 360.0
            self._elevation = float(min(80.0, max(8.0, elevation)))

    def frame(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._frame)

    def start(self, mode: str, pace: float, task_id: str | None = None) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._stop.set()
                previous = self._thread
            else:
                previous = None
            self._stop = threading.Event()
            stop = self._stop
            self._frame = {**_blank(mode), "status": "running", "scene_id": task_id}
        if previous is not None:
            previous.join(timeout=2.0)

        def publish(frame: dict[str, Any]) -> None:
            with self._lock:
                self._frame = frame

        def worker() -> None:
            try:
                if mode == "view":
                    if not task_id:
                        raise ValueError("view needs a task_id")
                    run_view(task_id, publish, stop=stop, aim=self.aim_now, catalog=self.catalog)
                else:
                    run_loop(mode, publish, pace=pace, stop=stop, aim=self.aim_now, catalog=self.catalog)
            except Exception as exc:  # noqa: BLE001 - surface it on the frame; the page shows it
                publish({**_blank(mode), "status": "done", "error": str(exc)})

        thread = threading.Thread(target=worker, name="rc-preview", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()


class _Handler(BaseHTTPRequestHandler):
    server: "PreviewServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send(200, {"ready": True})
        elif path == "/frame":
            self._send(200, self.server.state.frame())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/start":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid JSON"})
            return
        if not isinstance(body, dict):
            self._send(400, {"error": "expected a JSON object"})
            return
        if "azimuth" in body and "elevation" in body and "mode" not in body:
            try:
                self.server.state.aim(float(body["azimuth"]), float(body["elevation"]))
            except (TypeError, ValueError):
                self._send(400, {"error": "azimuth and elevation must be numbers"})
                return
            self._send(200, {"ok": True})
            return
        mode = body.get("mode")
        if mode not in ("public", "hidden", "contest", "view"):
            self._send(400, {"error": "mode must be public, hidden, contest, or view"})
            return
        task_id = body.get("task_id")
        if self.server.public_only and (mode != "view" or task_id not in PUBLIC_IDS):
            self._send(400, {"error": "preview requires a public task id"})
            return
        task_ids = self.server.state.catalog.defaults
        if mode == "view" and task_id not in task_ids:
            self._send(400, {"error": "unknown task id"})
            return
        if mode in ("hidden", "contest") and not any(task_id not in PUBLIC_IDS for task_id in task_ids):
            self._send(400, {"error": "organizer mode requires a task pack"})
            return
        try:
            pace = float(body.get("pace", 0.3))
        except (TypeError, ValueError):
            self._send(400, {"error": "pace must be a number"})
            return
        if pace < 0:
            pace = 0.0
        try:
            self.server.state.start(mode, pace, None if mode != "view" else str(task_id))
        except RuntimeError as exc:
            self._send(409, {"error": str(exc)})
            return
        self._send(200, {"ok": True, "mode": mode})


class PreviewServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: PreviewState, *, public_only: bool = False) -> None:
        super().__init__(address, _Handler)
        self.state = state
        self.public_only = public_only


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rc_engine.preview")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--public-only", action="store_true")
    modes.add_argument("--task-packs", help="directory of organizer task packs")
    args = parser.parse_args(argv)
    catalog = builtin_catalog()
    if args.task_packs:
        try:
            load_packs(args.task_packs, catalog)
        except PackError as exc:
            parser.error(str(exc))
    server = PreviewServer((args.host, args.port), PreviewState(catalog), public_only=args.public_only)
    print(f"[rc_preview] listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
