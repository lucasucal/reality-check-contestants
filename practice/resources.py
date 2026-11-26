"""Resource summary printed at the end of a practice run (CS-AC-8): load time, peak memory where it can be
measured, /act latency percentiles and /act time per repetition against the judged 90 s budget.

Measurement only: nothing here changes how the policy is called or limits it. Numbers are this machine's,
not the judging hardware's.
"""
from __future__ import annotations

import math
import shutil
import statistics
import threading
import time
from typing import Any, Callable, Optional

from rc_runner.policy_client import PolicyClient

JUDGED_ACT_BUDGET_SECONDS = 90.0
WARMUP_PREFIX = "warmup-"
GIB = 1024 ** 3


def format_memory(n: float) -> str:
    """Bytes as text teams read: whole MiB below 1 GiB (a small peak never reads 0.0 GiB), else one-decimal GiB."""
    if n < GIB:
        return f"{round(n / 1024 ** 2)} MiB"
    return f"{n / GIB:.1f} GiB"


class TimedPolicyClient(PolicyClient):
    """PolicyClient that records the wall time of every /act call (all attempts included) per episode."""

    def __init__(self, *args: Any, clock: Callable[[], float] = time.monotonic, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._clock = clock
        self._lock = threading.Lock()
        self.calls: list = []  # (episode_id, start, seconds, ok)

    def act(self, body: bytes, *, episode_id: str, **kwargs: Any) -> Any:
        start = self._clock()
        ok = False
        try:
            result = super().act(body, episode_id=episode_id, **kwargs)
            ok = True
            return result
        finally:
            with self._lock:
                self.calls.append((episode_id, start, self._clock() - start, ok))


def percentile(values: list, q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class MemorySampler:
    """Peak container memory for local practice: the runner's own measurement (cgroup working set plus the GPU
    memory of the container's processes), with an unreachable limit and no kill."""

    def __init__(self, container: str, *, gpu: bool, docker_bin: str = "docker", interval: float = 1.0) -> None:
        from rc_runner.memwatch import MemoryWatchdog
        self.gpu = bool(gpu and shutil.which("nvidia-smi"))
        self.notes: list = []
        self._dog = MemoryWatchdog(container=container, limit_bytes=1 << 62, interval=interval, kill=lambda: None,
                                   log=self.notes.append, docker_bin=docker_bin, check_gpu=self.gpu,
                                   outage_seconds=float("inf"))

    def start(self) -> "MemorySampler":
        self._dog.start()
        return self

    def stop(self) -> None:
        self._dog.stop()

    def describe(self) -> str:
        d = self._dog
        peak = max(d.peak, d.read_cgroup_peak() if not self.gpu else 0)
        if d.samples == 0 and not peak:
            return "peak memory: not measured (container cgroup not readable on this host)"
        if self.gpu:
            return (f"peak memory: {format_memory(d.peak)} combined (CPU {format_memory(d.peak_cpu)} + GPU "
                    f"{format_memory(d.peak_gpu)}, sampled every {d.interval:g} s)")
        return f"peak memory: {format_memory(peak)} CPU (cgroup; GPU memory not measured on this host)"


def _row(key: str, value: str) -> str:
    """"  peak memory   3.0 GiB ...": the summary's label column, so the values line up."""
    return f"  {key:<13} {value}"


def summary(calls: list, *, server_started: Optional[float], memory_line: str, memory_limit: str,
            transport_note: Optional[str] = None, budget_ended: int = 0) -> list:
    """Lines of the resource summary. `calls` are TimedPolicyClient.calls; `budget_ended` counts repetitions that
    ended because their /act budget ran out (--judged runs; the call cut at the deadline is not in `calls`)."""
    out = ["Resource summary (this machine, not the judging hardware):"]
    warm = [c for c in calls if str(c[0]).startswith(WARMUP_PREFIX)]
    first = min((c[1] for c in warm), default=None)
    if server_started is not None and first is not None:
        out.append(_row("load", f"ready after {first - server_started:.1f} s (policy start to the first warmup /act; "
                                "judged startup plus warmup allows 1800 s per policy start, paid again for each task)"))
    else:
        out.append(_row("load", "not measured (the policy never reached warmup)"))
    memory = memory_line[len("peak memory: "):] if memory_line.startswith("peak memory: ") else memory_line
    out.append(_row("peak memory", f"{memory} (judged limit {memory_limit} combined CPU + GPU)"))
    episodes = [c for c in calls if not str(c[0]).startswith(WARMUP_PREFIX)]
    if not episodes:
        out.append(_row("/act calls", "none during repetitions"))
        return out
    ms = [c[2] * 1000 for c in episodes]
    out.append(_row("/act calls", f"{len(ms)} calls, p50 {percentile(ms, 50):.1f} ms, p95 {percentile(ms, 95):.1f} ms, "
                                  f"max {max(ms):.1f} ms" + (f" ({transport_note})" if transport_note else "")))
    per_rep: dict = {}
    for episode_id, _start, seconds, _ok in episodes:
        per_rep[episode_id] = per_rep.get(episode_id, 0.0) + seconds
    totals = sorted(per_rep.values())
    out.append(_row("/act per rep", f"median {statistics.median(totals):.1f} s, max {totals[-1]:.1f} s of the "
                                    f"{JUDGED_ACT_BUDGET_SECONDS:g} s judged budget ({len(totals)} repetitions)"))
    over = max(budget_ended, sum(1 for t in totals if t >= JUDGED_ACT_BUDGET_SECONDS))
    if over:
        out.append(f"  WARNING: {over} repetition(s) ran out of the {JUDGED_ACT_BUDGET_SECONDS:g} s /act budget; at this "
                   "speed a judged repetition ends early when its /act budget runs out"
                   + (" (here transport time counts too)" if transport_note else ""))
    return out
