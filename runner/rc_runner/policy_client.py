"""HTTP client for the team container's POST /act (docs/platform.md §6.1, plan §4).

Retry discipline (contract-review #2): on timeout/connect failure retry AT MOST
once with the byte-identical body (the pure-function contract makes that safe).
Echo discipline: the response must echo {episode_id, step} in its body and headers.
"""
from __future__ import annotations

import json
import math
import base64
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


class _NonFinite(ValueError):
    pass


class _Overflow(ValueError):
    pass


def _finite_constant(name: str) -> float:  # NaN / Infinity / -Infinity literals
    raise _NonFinite(name)


def _finite_float(number: str) -> float:
    value = float(number)
    if not math.isfinite(value):
        raise _Overflow(number[:20])  # a finite-looking literal such as 1e400
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field')
        result[key] = value
    return result


DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # decoded /act body cap (RC_POLICY_RESPONSE_MAX_BYTES)


class _TooLarge(Exception):
    def __init__(self, read: int) -> None:
        self.read = read


def _size_text(n: int) -> str:
    return f"{n // 1048576} MiB" if n >= 1048576 and n % 1048576 == 0 else f"{n} bytes"


class PolicyFailure(Exception):
    """An episode-level team failure with its plan §5 code."""

    def __init__(self, code: str, message: str, attempts: list[dict[str, Any]] | None = None,
                 team_detail: str | None = None, reason: str | None = None, held_reason: str | None = None) -> None:
        super().__init__(message)
        # Set for a malformed (but delivered) 200 response: the episode loop answers it with a hold-position
        # step instead of ending the episode. One of motion.HOLD_REASONS.
        self.held_reason = held_reason
        self.team_detail = team_detail
        self.reason = reason  # error-surfacing contract FAILURE_REASONS value, or None
        self.code = code
        self.message = message
        self.attempts = attempts or []


@dataclass
class ActResult:
    action: Any
    attempt: int  # 1, or 2 when the single retry was needed
    total_ms: float
    failed_attempts: list[dict[str, Any]] = field(default_factory=list)
    request_seconds: float = 0.0
    ignored_fields: list[str] = field(default_factory=list)  # unknown top-level keys of the /act response


class PolicyClient:
    def __init__(self, base_url: str, *, act_timeout: float, connect_timeout: float, retries: int = 1,
                 transport: httpx.BaseTransport | None = None,
                 container_alive: Callable[[], bool] | None = None,
                 max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES) -> None:
        self._max_response = max(int(max_response_bytes), 1)
        self._http = httpx.Client(base_url=base_url, transport=transport, timeout=httpx.Timeout(act_timeout, connect=connect_timeout))
        self._act_timeout = act_timeout
        self._connect_timeout = connect_timeout
        self._retries = max(retries, 0)
        self._alive = container_alive

    def close(self) -> None:
        self._http.close()

    def get(self, path: str) -> httpx.Response:
        return self._http.get(path, timeout=httpx.Timeout(5.0))

    def act(self, body: bytes, *, episode_id: str, step: int, token: str, timeout: float | None = None,
            retries: int | None = None, record: Callable[[dict[str, Any]], None] | None = None,
            request_finished: Callable[[float], None] | None = None,
            request_started: Callable[[float], None] | None = None) -> ActResult:
        """POST the envelope bytes; returns the verified action or raises PolicyFailure."""
        headers = {
            "Content-Type": "application/json",
            "X-RC-Episode-Id": episode_id,
            "X-RC-Step": str(step),
            "X-RC-Episode-Token": token,
        }
        timeout_cfg = httpx.Timeout(timeout, connect=min(timeout, self._connect_timeout)) if timeout is not None else None
        max_retries = self._retries if retries is None else retries
        failed: list[dict[str, Any]] = []
        for attempt in range(1, max_retries + 2):
            start = time.monotonic()
            if request_started:
                request_started(start)
            try:
                with self._http.stream("POST", "/act", content=body, headers=headers,
                                       **({"timeout": timeout_cfg} if timeout_cfg else {})) as response:
                    status, resp_headers = response.status_code, response.headers
                    payload = self._read_capped(response)
            except _TooLarge as big:
                request_seconds = time.monotonic() - start
                if request_finished:
                    request_finished(request_seconds)
                if record:
                    record({'attempt': attempt, 'error': 'response_too_large', 'body_bytes_read': big.read,
                            'limit_bytes': self._max_response})
                raise PolicyFailure(
                    "policy_protocol_violation",
                    f"/act response exceeded the {self._max_response}-byte limit (/act responses are small JSON "
                    "actions; do not return large payloads)", failed,
                    team_detail=f"Your /act response was larger than {_size_text(self._max_response)}; /act returns small JSON actions, so do not put "
                                "large payloads (images, debug dumps) in it.", reason="too_large") from None
            except httpx.TransportError as exc:  # timeout / connect / read / reset: retryable
                if request_finished:
                    request_finished(time.monotonic() - start)
                if record:
                    record({'attempt': attempt, 'error': repr(exc)})
                failed.append({"attempt": attempt, "error": type(exc).__name__, "ms": round((time.monotonic() - start) * 1000, 1)})
                continue
            request_seconds = time.monotonic() - start
            if request_finished:
                request_finished(request_seconds)
            total_ms = round(request_seconds * 1000, 1)
            if record:
                record({'attempt': attempt, 'status': status, 'headers': dict(resp_headers),
                        'body_base64': base64.b64encode(payload).decode('ascii')})
            if status >= 400:
                raise PolicyFailure("policy_http_error", f"/act returned HTTP {status}: {payload[:200].decode('utf-8', 'replace')}", failed,
                                    reason="act_http_error")
            ignored: list[str] = []
            action = self._verify(payload, resp_headers, episode_id, step, failed, ignored)
            return ActResult(action, attempt, total_ms, failed, request_seconds, ignored_fields=ignored)
        crashed = self._alive is not None and not self._alive()
        code = "policy_crashed" if crashed else "policy_timeout"
        detail = "policy container died" if crashed else f"/act timed out or was unreachable after {len(failed)} attempts"
        raise PolicyFailure(code, detail, failed, reason=None if crashed else "act_timeout")

    def _read_capped(self, response: httpx.Response) -> bytes:
        """Stream the decoded body; refuse (without buffering the rest) once it passes the cap."""
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_response and not response.headers.get("content-encoding"):
            raise _TooLarge(0)
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self._max_response:
                raise _TooLarge(size)
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _verify(payload: bytes, headers: Any, episode_id: str, step: int, failed: list[dict[str, Any]],
                ignored: list[str] | None = None) -> Any:
        try:
            data = json.loads(payload.decode("utf-8"), parse_float=_finite_float, parse_constant=_finite_constant,
                              object_pairs_hook=_unique_object)
        except _NonFinite:
            raise PolicyFailure("policy_protocol_violation", "/act response must be valid JSON with finite numbers",
                                failed, reason="not_json", held_reason="non_finite") from None
        except _Overflow:
            raise PolicyFailure("policy_protocol_violation", "/act response must be valid JSON with finite numbers",
                                failed, reason="not_json", held_reason="overflow") from None
        except ValueError as exc:
            overflow = "Exceeds the limit" in str(exc)  # a huge integer literal
            raise PolicyFailure("policy_protocol_violation",
                                "/act response must be valid JSON with finite numbers", failed, reason="not_json",
                                held_reason="overflow" if overflow else "invalid_json") from None
        if not isinstance(data, dict):
            raise PolicyFailure("policy_protocol_violation", "/act response must be a JSON object", failed,
                                reason="not_json", held_reason="invalid_json")
        echoed_step = data.get("step")
        if data.get("episode_id") != episode_id or type(echoed_step) is not int or echoed_step != step:
            raise PolicyFailure(
                "policy_protocol_violation",
                f"echo mismatch: sent episode_id={episode_id!r} step={step}, "
                f"got episode_id={data.get('episode_id')!r} step={echoed_step!r}", failed, reason="echo_mismatch", held_reason="echo_mismatch")
        if headers.get("X-RC-Episode-Id") != episode_id or headers.get("X-RC-Step") != str(step):
            raise PolicyFailure("policy_protocol_violation",
                                "/act must echo the request in X-RC-Episode-Id and X-RC-Step", failed,
                                reason="echo_mismatch", held_reason="echo_mismatch")
        if "action" not in data:
            raise PolicyFailure("policy_protocol_violation", "/act response has no action", failed, reason="no_action",
                                held_reason="missing_joint_delta")
        if ignored is not None:
            from .motion import note_ignored
            note_ignored(ignored, [k for k in data if k not in ("episode_id", "step", "action")])
        return data["action"]
