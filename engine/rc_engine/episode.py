"""The one in-flight episode: ids, step counter, validation, response bodies (§2 of the engine handoff)."""
from __future__ import annotations

import inspect
import uuid
import math
import time
from typing import Any

from .cameras import CAMERA_SETS
from .env import Env
from .schema import ProtocolViolation, encode_obs, encode_views, goal_jpeg, parse_action

MAX_SEED = 2**32


class EngineError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class EpisodeManager:
    def __init__(self, envs: dict[str, Env], jpeg_quality: int) -> None:
        self.envs = envs
        self.jpeg_quality = jpeg_quality
        self.env: Env | None = None
        self.episode_id: str | None = None
        self.done = False
        self.last_payload: dict[str, Any] | None = None

    def reset(self, body: dict[str, Any]) -> dict[str, Any]:
        task_id, seed = body.get("task_id"), body.get("seed")
        if not isinstance(task_id, str):
            raise EngineError(400, "bad_request", "task_id must be a string")
        if task_id not in self.envs:
            raise EngineError(400, "unknown_task", f"unknown task_id {task_id!r}; see GET /tasks")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED:
            raise EngineError(400, "bad_request", f"seed must be an integer in [0, {MAX_SEED}), got {seed!r}")
        self.env, self.episode_id, self.done = None, None, False
        env = self.envs[task_id]
        config = body.get("camera_configuration", "three_view")
        if config not in CAMERA_SETS:
            known = ", ".join(sorted(CAMERA_SETS))
            raise EngineError(400, "bad_request", f"camera_configuration must be one of {known}")
        max_steps = body.get("max_steps", "legacy")
        if max_steps != "legacy" and max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
        ):
            raise EngineError(400, "bad_request", "max_steps must be null or a positive integer")
        # Per-episode options (fastener tasks, tasks/variation.py): {"profile": "public"|"eval", "nudge": bool}.
        options = body.get("options")
        if options is not None:
            if "options" not in inspect.signature(env.reset).parameters:
                raise EngineError(400, "bad_request", f"task {task_id!r} takes no reset options")
            from .tasks.variation import parse_options  # noqa: PLC0415 - MuJoCo-free engines never get here

            try:
                profile, nudge = parse_options(options)
            except ValueError as exc:
                raise EngineError(400, "bad_request", str(exc)) from exc
            options = {"profile": profile, "nudge": nudge}
        configure = getattr(env, "configure_episode", None)
        if configure is not None:
            poses = body.get("camera_poses")
            if poses is not None and not isinstance(poses, dict):
                raise EngineError(400, "bad_request", "camera_poses must be an object")
            try:
                configure(config, None if max_steps == "legacy" else (0 if max_steps is None else max_steps), poses)
            except (ValueError, TypeError) as exc:
                raise EngineError(400, "bad_request", str(exc)) from exc
        try:
            raw = env.reset(seed) if options is None else env.reset(seed, options=options)  # type: ignore[call-arg]
            observation = encode_obs(env.spec, 0, raw, self.jpeg_quality)
        except Exception as exc:
            raise EngineError(500, "engine_error", f"reset failed: {exc!r}") from exc
        self.env, self.episode_id = env, uuid.uuid4().hex
        self.last_payload = self._with_views(
            {"episode_id": self.episode_id, "step": 0, "observation": observation, "done": False}, env)
        return self.last_payload

    def step(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply one action or chunk. For tasks that measure motion limits (arm_limits.py), `info` carries
        `speed_scaled` {"tcp_speed": n, "elbow_speed": m}: how many of the applied substeps went over the 1.5 m/s
        TCP / elbow speed and were scaled down; and `safety_peaks` {"tcp_force_n", "momentum_kg_m_s", "power_w"}:
        the repetition's running peaks so far (measured, never enforced)."""
        if "episode_id" not in body or "action" not in body:
            raise EngineError(400, "bad_request", "body must contain episode_id and action")
        if self.env is None or body["episode_id"] != self.episode_id:
            raise EngineError(409, "unknown_episode", f"episode {body['episode_id']!r} is not the current episode; call /reset")
        if self.done:
            raise EngineError(409, "episode_done", f"episode {self.episode_id} is done; call /reset")
        expected = body.get("step", self.env.step_count)
        if isinstance(expected, bool) or expected != self.env.step_count:
            raise EngineError(409, "step_mismatch", f"request is for step {expected!r}; episode is at step {self.env.step_count}")
        try:
            actions = parse_action(body["action"])
        except ProtocolViolation as exc:
            raise EngineError(400, "policy_protocol_violation", str(exc)) from exc
        env = self.env
        deadline = body.get('deadline_monotonic')
        if deadline is not None:
            try:
                valid_deadline = not isinstance(deadline, bool) and isinstance(deadline, (int, float)) and math.isfinite(deadline)
            except OverflowError:
                valid_deadline = False
            if not valid_deadline:
                raise EngineError(400, 'bad_request', 'deadline_monotonic must be a finite number')
        try:
            hashes = []
            scaled = {"tcp_speed": 0, "elbow_speed": 0}
            peaks = None
            for action in actions:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                result = env.step(action)
                hashes.append(env.state_hash())
                if result.limits:
                    for key in scaled:
                        scaled[key] += int(result.limits["speed_scaled"][key])
                    peaks = result.limits["safety_peaks"]
                if result.done:
                    break
            if not hashes:
                assert self.last_payload is not None
                return {**self.last_payload, 'info': {'applied': 0, 'substep_hashes': [], 'deadline_expired': True}}
            observation = encode_obs(env.spec, env.step_count, result.obs, self.jpeg_quality)
        except Exception as exc:
            self.env, self.episode_id = None, None
            raise EngineError(500, "engine_error", f"step failed, episode dropped: {exc!r}") from exc
        self.done = result.done
        self.last_payload = self._with_views({
            "episode_id": self.episode_id,
            "step": env.step_count,
            "observation": observation,
            "done": result.done,
            "success": bool(result.success),
            "score": float(result.score),
            "doneness": float(result.doneness),
            "speed": float(result.speed),
            "energy": float(result.energy),
            "quality": float(result.quality),
            "quality_limit": result.quality_limit,
            "info": {"state_hash": hashes[-1], "substep_hashes": hashes, "applied": len(hashes),
                     "deadline_expired": deadline is not None and time.monotonic() >= deadline,
                     **({"speed_scaled": scaled, "safety_peaks": peaks} if peaks is not None else {})},
        }, env)
        return self.last_payload

    def render(self, body: dict[str, Any]) -> dict[str, Any]:
        """Organizer replay (`POST /render`, only when RC_ENGINE_REPLAY=1): re-render the policy cameras
        and the goal image from a recorded simulator snapshot. Never part of an episode."""
        task_id, snapshot = body.get("task_id"), body.get("simulator_state")
        if not isinstance(task_id, str) or task_id not in self.envs:
            raise EngineError(400, "unknown_task", f"unknown task_id {task_id!r}")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("state"), list):
            raise EngineError(400, "bad_request", "simulator_state must be a recorded snapshot object")
        env = self.envs[task_id]
        renderer = getattr(env, "render_snapshot", None)
        configure = getattr(env, "configure_episode", None)
        if renderer is None or configure is None:
            raise EngineError(400, "bad_request", f"task {task_id!r} cannot render from a snapshot")
        config = body.get("camera_configuration", "three_view")
        if config not in CAMERA_SETS:
            raise EngineError(400, "bad_request", "unknown camera_configuration")
        try:
            configure(config, None, body.get("camera_poses"))
            frames = renderer(snapshot)
        except (ValueError, TypeError, KeyError) as exc:
            raise EngineError(400, "bad_request", str(exc)) from exc
        self.env, self.episode_id = None, None  # the render overwrote the live data
        return {"task_id": task_id, "images": encode_views(frames, self.jpeg_quality),
                "goal": goal_jpeg(env.spec.goal_id(), self.jpeg_quality)}

    def _with_views(self, payload: dict[str, Any], env: Env) -> dict[str, Any]:
        snapshot = getattr(env, "replay_snapshot", None)
        if snapshot is not None:
            payload["simulator_state"] = snapshot()
        frames = getattr(env, "audience_images", None)
        if frames:
            payload["views"] = encode_views(frames, self.jpeg_quality)
        return payload
