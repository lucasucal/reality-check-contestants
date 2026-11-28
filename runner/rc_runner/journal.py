"""JSONL journals (plan §6): one file per episode, one line per control step.

Journals stay on the runner host under RC_WORK_DIR/journals/{submission}/; the
backend only learns the `runner://{hostname}/journals/{submission}/` locator.
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import queue
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

JOURNAL_FORMAT = 3  # Per-role goal digests alongside current-image digests.


def now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def journal_locator(submission_id: int) -> str:
    """The journal_path reported to the backend: a host-qualified runner locator."""
    return f"runner://{socket.gethostname()}/journals/{submission_id}/"


def episode_filename(ordinal: int) -> str:
    return f"episode-{ordinal}.jsonl"


def state_chain_digest(substep_hashes: list[str]) -> str:
    """sha256 hex over the \\n-joined substep hashes of the whole episode (plan §6)."""
    return hashlib.sha256("\n".join(substep_hashes).encode()).hexdigest()


def parse_journal_images(spec: str) -> int:
    """RC_JOURNAL_IMAGES -> keep a full image body every N steps: 0 = never (default), 1 = all."""
    text = (spec or "none").strip().lower()
    if text == "none":
        return 0
    if text == "all":
        return 1
    if text.startswith("every:"):
        try:
            n = int(text[6:])
        except ValueError:
            n = 0
        if n >= 1:
            return n
    raise ValueError(f"journal images must be none, all or every:N (N >= 1), got {spec!r}")


def compact_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Engine `simulator_state` with its float list packed exactly: float64 little-endian, zlib, base64.
    Lossless on purpose: a replay must reproduce the original pixels bit for bit."""
    out = {k: v for k, v in snapshot.items() if k != "state"}
    state = snapshot.get("state")
    if isinstance(state, list):
        raw = struct.pack(f"<{len(state)}d", *state)
        out["state_z"] = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
        out["state_n"] = len(state)
    return out


def expand_state(compact: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `compact_state`: a snapshot dict the engine can render from."""
    out = {k: v for k, v in compact.items() if k not in ("state_z", "state_n")}
    if "state_z" in compact:
        raw = zlib.decompress(base64.b64decode(compact["state_z"]))
        out["state"] = list(struct.unpack(f"<{len(raw) // 8}d", raw))
    return out


def image_digest(b64: str) -> dict[str, Any]:
    """sha256 and byte size of the decoded JPEG behind one base64 image string."""
    raw = base64.b64decode(b64)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def image_digests(images: Any) -> dict[str, Any]:
    if not isinstance(images, dict):
        return {}
    return {name: image_digest(value) for name, value in images.items() if isinstance(value, str)}


def evidence_response(response: dict[str, Any]) -> dict[str, Any]:
    """An engine /reset or /step response as journaled by default: no image bodies.
    Policy images become digests, audience `views` are dropped, simulator_state is packed."""
    out = {k: v for k, v in response.items() if k not in ("views", "simulator_state")}
    observation = response.get("observation")
    if isinstance(observation, dict):
        obs = {k: v for k, v in observation.items() if k not in ("images", "goal")}
        obs["image_digests"] = image_digests(observation.get("images"))
        if isinstance(observation.get("goal"), dict):
            obs["goal_digests"] = image_digests(observation["goal"])
        elif isinstance(observation.get("goal"), str):
            obs["goal_digest"] = image_digest(observation["goal"])
        out["observation"] = obs
    if isinstance(response.get("simulator_state"), dict):
        out["simulator_state"] = compact_state(response["simulator_state"])
    return out


class JournalWriter:
    """Bounded background JSONL writer. A crash may lose the queued suffix."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO | None = path.open("w", encoding="utf-8")
        self._queue: queue.Queue = queue.Queue(maxsize=32)
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                if self._error is None:
                    assert self._fh is not None
                    self._fh.write(json.dumps(record, separators=(',', ':')) + '\n')
                    self._fh.flush()
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is not None:
            if self._error:
                raise OSError(f'journal write failed: {self._error}') from self._error
            self._queue.put(record)

    def close(self) -> None:
        if self._fh is not None:
            self._queue.put(None)
            self._thread.join()
            try:
                self._fh.close()
            finally:
                self._fh = None
            if self._error:
                raise OSError(f'journal write failed: {self._error}') from self._error
