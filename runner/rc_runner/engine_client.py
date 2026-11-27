"""HTTP client for the engine (engine/README.md: /healthz /readyz /tasks /reset /step).

Error policy (plan §5): anything the engine does wrong (connection failure,
timeout, 5xx incl. engine_error/not_ready, 409 step_mismatch/unknown_episode/...)
is EngineInfraError — never the team's fault. The one team-attributable error is
400 policy_protocol_violation on /step (a rejected action), ActionRejected.
Specs are read from /tasks; nothing about joint counts is hardcoded here.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from .errors import ActionRejected, EngineInfraError


class EngineClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=5.0), transport=transport)

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, body: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        try:
            kwargs = {"timeout": httpx.Timeout(timeout, connect=min(timeout, 5.0))} if timeout is not None else {}
            response = self._http.request(method, path, json=body, **kwargs) if body is not None else self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise EngineInfraError(f"engine {method} {path} failed: {exc!r}") from exc
        data = _json(response)
        if response.status_code >= 400:
            err = data.get("error") if isinstance(data.get("error"), dict) else {}
            code, message = err.get("code", "?"), err.get("message", response.text[:300])
            if response.status_code == 400 and code == "policy_protocol_violation" and path == "/step":
                raise ActionRejected(str(message))
            raise EngineInfraError(f"engine {method} {path} -> {response.status_code} {code}: {message}")
        return data

    def healthz(self) -> bool:
        try:
            return self._http.get("/healthz").status_code == 200
        except httpx.HTTPError:
            return False

    def readyz(self) -> dict[str, Any]:
        """{ready, tasks, schema_version, scorer[, error]} from the engine; a connection failure or
        non-200 is {ready: False, _unreachable: reason} (no `error` key: the engine's own
        `error` means its task load failed for good)."""
        try:
            response = self._http.get("/readyz")
            data = _json(response)
        except httpx.HTTPError as exc:
            return {"ready": False, "_unreachable": repr(exc)}
        if response.status_code != 200:
            return {"ready": False, "_unreachable": f"HTTP {response.status_code}"}
        return data

    def tasks(self) -> tuple[int | None, list[dict[str, Any]]]:
        data = self._request("GET", "/tasks")
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(t, dict) and isinstance(t.get("task_id"), str) for t in tasks):
            raise EngineInfraError("engine /tasks returned an unexpected shape")
        return data.get("schema_version"), tasks

    def reset(self, task_id: str, seed: int, **options: Any) -> dict[str, Any]:
        data = self._request("POST", "/reset", {"task_id": task_id, "seed": seed, **options})
        if not isinstance(data.get("episode_id"), str) or not isinstance(data.get("observation"), dict):
            raise EngineInfraError("engine /reset returned an unexpected shape")
        return data

    def step(self, episode_id: str, step: int, action: Any, *, timeout: float | None = None,
             deadline_monotonic: float | None = None, render_pose: bool = False) -> dict[str, Any]:
        body = {"episode_id": episode_id, "action": action, "step": step}
        if deadline_monotonic is not None:
            body['deadline_monotonic'] = deadline_monotonic
        if render_pose:
            body["render_pose"] = True
        data = self._request("POST", "/step", body, timeout=timeout)
        if "observation" not in data or "done" not in data or not isinstance(data.get("info"), dict):
            raise EngineInfraError("engine /step returned an unexpected shape")
        return data


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
