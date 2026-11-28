"""The judged episode plan: the seed and engine options of every judged episode, per task.

Built from two inputs only: the event's secret judged seed (claim `eval.seeds[0]`, drawn once per event into
`event_secrets`) and the engine `task_id`. So every team, submission, re-run, re-score and replay gets the same
plan for as long as the secret is unchanged. The plan is organizer-only: it is journaled and kept in the
organizer `episodes` evidence, never in team-facing output.

Scheme v1. For task T, episode index i in 0..n-1 (n = `eval.repetitions`, 4 by default) and a tag:

    H(tag, i) = int.from_bytes(sha256(f"rc-judged-v1:{secret}:{T}:{tag}:{i}").digest()[:4], "big") & 0x7fffffff

- seed_i = H("seed", i). A seed already used by an earlier episode of T is drawn again with tag "seed.1",
  "seed.2", ... so the n seeds of a task are distinct (a repeat has probability about n^2 / 2^32).
- profile: the one episode with the smallest H("eval", i) uses profile "eval"; the others "public".
Ties fall back to the lower index.
"""
from __future__ import annotations

import hashlib
from typing import Any

SCHEME = "rc-judged-v1"
SEED_MASK = 0x7FFFFFFF  # below 2**31, like the secret itself: valid for the engine and exact in JSON/JS
DEFAULT_OPTIONS: dict[str, Any] = {"profile": "public"}


def _draw(secret: int, task_id: str, tag: str, index: int) -> int:
    digest = hashlib.sha256(f"{SCHEME}:{secret}:{task_id}:{tag}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & SEED_MASK


def task_plan(secret: int, task_id: str, episodes: int = 4) -> list[tuple[int, dict[str, Any]]]:
    """[(seed, {"profile": ...}), ...] for the judged episodes of one task, in run order."""
    seeds: list[int] = []
    for i in range(episodes):
        seed, retry = _draw(secret, task_id, "seed", i), 0
        while seed in seeds:
            retry += 1
            seed = _draw(secret, task_id, f"seed.{retry}", i)
        seeds.append(seed)
    eval_index = min(range(episodes), key=lambda i: (_draw(secret, task_id, "eval", i), i)) if episodes else -1
    return [(seed, {"profile": "eval" if i == eval_index else "public"})
            for i, seed in enumerate(seeds)]


def engine_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """The `options` object for the engine /reset body, or None when every option is the engine default
    (then the field is left out, so an engine that predates per-episode options runs the episode unchanged)."""
    merged = {**DEFAULT_OPTIONS, **(options or {})}
    return merged if merged != DEFAULT_OPTIONS else None
