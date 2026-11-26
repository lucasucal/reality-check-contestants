"""Policy memory watchdog (docs/policy-runtime-pin.md section 7).

On the GB10 the container memory cgroup does not charge plain `cudaMalloc` allocations, so
`--memory` only bounds CPU-side memory. The policy's total memory is therefore measured as
cgroup working set (memory.current minus inactive file cache, as `docker stats` reports it)
plus the GPU memory of the host PIDs inside the container's cgroup (`nvidia-smi
--query-compute-apps`). When the total exceeds RC_POLICY_MEMORY the container is killed and the
verdict is exposed to the worker, which fails the attempt with `policy_memory_exceeded`.
Telemetry must be healthy: if the cgroup or nvidia-smi cannot be read for longer than
`outage_seconds` (including from startup) the watchdog records `outage` and the worker fails the
evaluation as a retryable infrastructure failure (`infra_error`, not charged to the team). A
silent fallback to a partial measurement would defeat the limit.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

GIB = 1024 ** 3
MIB = 1024 ** 2
FAILURE_CODE = "policy_memory_exceeded"
# Known issue for every policy_memory_exceeded (watchdog trip or kernel OOM kill), attached directly: a docker
# kill can be inspected before the container has exited (exit code not yet 137), so log/exit matching misses it.
KNOWN_ISSUE = "KSI-008"
OUTAGE_SECONDS = 10.0
NVIDIA_QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
_UNITS = {"": 1, "b": 1, "k": 1024, "m": MIB, "g": GIB, "t": 1024 ** 4}


def parse_memory_limit(text: str) -> int:
    """Docker-style size ('100g', '512m', '1.5g', '2gb') to bytes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([bkmgt]?)b?\s*", text.lower())
    if not m:
        raise ValueError(f"unparseable memory size {text!r}")
    return int(float(m.group(1)) * _UNITS[m.group(2)])


def parse_gpu_usage(output: str) -> dict[int, int]:
    """`pid, used_memory(MiB)` CSV lines to {pid: bytes}; [N/A] and junk lines are skipped."""
    usage: dict[int, int] = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = usage.get(int(parts[0]), 0) + int(parts[1]) * MIB
    return usage


def _gib(n: float) -> str:
    return f"{n / GIB:.1f} GiB"


_LIMIT_RE = re.compile(r"\AC[^\n]{0,40}?memory limit (\d{1,5}\.\d) GiB[,.;]")
_PEAK_RE = re.compile(r"\bmeasured peak (\d{1,5}\.\d) GiB\b")


def bounded_detail(detail: Optional[str]) -> Optional[str]:
    """H3 / BAD-18: the only team detail a hidden-phase memory failure may carry: the limit and the measured
    peak (GiB, one decimal), re-rendered from the two numbers. None when the detail has no limit clause."""
    limit = _LIMIT_RE.search(detail or "")
    if not limit:
        return None
    peak = _PEAK_RE.search(detail or "")
    return f"Combined CPU + GPU memory limit {limit.group(1)} GiB" + (f", measured peak {peak.group(1)} GiB." if peak else ".")


class Verdict:
    def __init__(self, total: int, cpu: int, gpu: int, limit: int) -> None:
        self.total, self.cpu, self.gpu, self.limit = total, cpu, gpu, limit

    @property
    def team_detail(self) -> str:
        return (f"Combined CPU + GPU memory limit {_gib(self.limit)}, measured peak {_gib(self.total)} "
                f"({_gib(self.cpu)} CPU + {_gib(self.gpu)} GPU).")

    @property
    def message(self) -> str:
        return (f"The policy exceeded the combined CPU + GPU memory limit of {_gib(self.limit)} "
                f"(RC_POLICY_MEMORY): peak measured {_gib(self.total)} "
                f"({_gib(self.cpu)} CPU + {_gib(self.gpu)} GPU). The container was stopped.")


class MemoryWatchdog:
    """Polls one policy container until stopped. `run`/`read` are injectable for tests:
    run(argv, timeout) -> (returncode, stdout); read(path) -> text."""

    def __init__(self, *, container: str, limit_bytes: int, interval: float = 1.0,
                 kill: Callable[[], None], log: Callable[[str], None] = print,
                 run: Optional[Callable[[Sequence[str], float], tuple[int, str]]] = None,
                 read: Optional[Callable[[str], str]] = None, docker_bin: str = "docker",
                 cgroup_root: str = "/sys/fs/cgroup", smi_timeout: float = 5.0,
                 outage_seconds: float = OUTAGE_SECONDS, check_gpu: bool = True,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.container, self.limit, self.interval = container, limit_bytes, max(interval, 0.05)
        self._kill, self._log = kill, log
        self._run = run or self._subprocess_run
        self._read = read or (lambda p: Path(p).read_text(encoding="utf-8"))
        self._docker, self._root, self._smi_timeout = docker_bin, cgroup_root.rstrip("/"), smi_timeout
        self._outage_seconds, self._check_gpu, self._clock = outage_seconds, check_gpu, clock
        self._unhealthy_since: Optional[float] = self._clock()  # healthy telemetry is required from startup
        self.outage: Optional[str] = None
        self.peak = 0
        self.peak_cpu = 0
        self.peak_gpu = 0
        self.samples = 0
        # The cgroup's own recorded high-water mark (memory.peak, cgroup v1 memory.max_usage_in_bytes), when the
        # kernel exposes one. Unlike the sampled peak it also covers a spike between two samples (STA-26).
        self.cgroup_peak = 0
        self.verdict: Optional[Verdict] = None
        self._cg: Optional[str] = None
        self._warned: set[str] = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _subprocess_run(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        res = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout

    def _warn_once(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._log(f"memory watchdog {self.container}: WARNING {text}")

    # -- sampling ------------------------------------------------------------------
    def _cgroup_dir(self) -> Optional[str]:
        if self._cg:
            return self._cg
        try:
            rc, out = self._run([self._docker, "inspect", "-f", "{{.Id}}", self.container], 10)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("inspect", f"cannot resolve container id: {exc!r}")
            return None
        cid = out.strip()
        if rc != 0 or not re.fullmatch(r"[0-9a-f]{12,64}", cid):
            return None
        for cand in (f"{self._root}/system.slice/docker-{cid}.scope", f"{self._root}/docker/{cid}"):
            try:
                self._read(cand + "/memory.current")
            except OSError:
                continue
            self._cg = cand
            return cand
        self._warn_once("cgroup", f"no cgroup directory found for {cid[:12]}; memory.current unavailable")
        return None

    def _cpu_bytes(self, cg: str) -> int:
        current = int(self._read(cg + "/memory.current").strip())
        try:  # reclaimable page cache is not "used": the kernel drops it before any OOM
            for line in self._read(cg + "/memory.stat").splitlines():
                if line.startswith("inactive_file "):
                    current -= int(line.split()[1])
                    break
        except (OSError, ValueError):
            pass
        return max(current, 0)

    def _read_peak(self, cg: str) -> int:
        for name in ("memory.peak", "memory.max_usage_in_bytes"):
            try:
                text = self._read(f"{cg}/{name}").strip()
            except Exception:  # noqa: BLE001 - optional file (older kernels, cgroup v1/v2 differences)
                continue
            if text.isdigit():
                return int(text)
        return 0

    def read_cgroup_peak(self) -> int:
        """Best effort: the cgroup's recorded peak now (e.g. right after an OOM kill, while the cgroup still
        exists), else the last one seen while sampling. 0 when the kernel never exposed one."""
        if self._cg:
            self.cgroup_peak = max(self.cgroup_peak, self._read_peak(self._cg))
        return self.cgroup_peak

    def _pids(self, cg: str) -> set[int]:
        return {int(x) for x in self._read(cg + "/cgroup.procs").split() if x.isdigit()}

    def _gpu_bytes(self, cg: str) -> Optional[int]:
        """GPU bytes of the container's PIDs, or None when the measurement is unavailable."""
        if not self._check_gpu:
            return 0
        try:
            pids = self._pids(cg)
            rc, out = self._run(NVIDIA_QUERY, self._smi_timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            self._warn_once("nvidia-smi", f"nvidia-smi unavailable ({exc!r})")
            return None
        if rc != 0:
            self._warn_once("nvidia-smi", f"nvidia-smi exited {rc}")
            return None
        return sum(b for pid, b in parse_gpu_usage(out).items() if pid in pids)

    def _telemetry_failed(self, what: str) -> None:
        now = self._clock()
        if self._unhealthy_since is None:
            self._unhealthy_since = now
        if self.outage is None and now - self._unhealthy_since >= self._outage_seconds:
            self.outage = (f"Memory telemetry ({what}) was unavailable for {self._outage_seconds:g}s, so the "
                           "policy memory limit could not be enforced. This is an infrastructure problem, "
                           "not a policy failure.")
            self._log(f"memory watchdog {self.container}: {self.outage}")

    def sample(self) -> Optional[Verdict]:
        """One measurement; kills the container on a breach. Returns the verdict when tripped."""
        cg = self._cgroup_dir()
        if cg is None:
            self._telemetry_failed("container cgroup")
            return None
        try:
            cpu = self._cpu_bytes(cg)
        except (OSError, ValueError):
            self._telemetry_failed("cgroup memory.current")
            return None
        gpu = self._gpu_bytes(cg)
        if gpu is None:
            self._telemetry_failed("nvidia-smi")
            return None
        self._unhealthy_since = None
        self.cgroup_peak = max(self.cgroup_peak, self._read_peak(cg))
        total = cpu + gpu
        self.samples += 1
        if total > self.peak:
            self.peak, self.peak_cpu, self.peak_gpu = total, cpu, gpu
        if total > self.limit and self.verdict is None:
            self.verdict = Verdict(self.peak, self.peak_cpu, self.peak_gpu, self.limit)
            self._log(f"memory watchdog {self.container}: {self.verdict.message}")
            try:
                self._kill()
            except Exception as exc:  # noqa: BLE001 - the verdict stands even if kill errors
                self._log(f"memory watchdog {self.container}: kill failed: {exc!r}")
            return self.verdict
        return None

    # -- lifecycle -------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.sample():
                    return
            except Exception as exc:  # noqa: BLE001 - never let tooling bugs fail a submission
                self._warn_once("loop", f"sample failed: {exc!r}")
            self._stop.wait(self.interval)

    def start(self) -> "MemoryWatchdog":
        self._thread = threading.Thread(target=self._loop, name=f"memwatch-{self.container}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
