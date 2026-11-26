"""Shared pinned_image launch logic (docs/runtime-interfaces.md section 6): the exact
policy `docker run` argv, the judged extra-args allowlist, image-reference selection
and host checks. Pure functions; no docker calls, so the practice launcher can import it.
"""
from __future__ import annotations

import os
import platform as _platform
import re
import shutil
import stat
from pathlib import Path
from typing import Any, Sequence

import rc_checks

# Policy traffic stays on the internal bridge managed by the host guard.
POLICY_NETWORK = "rc-policy-internal"
POLICY_PORT = 8080  # the runtime image always serves here; Config.policy_port is the one place code reads it
SUBMISSION_UID_GID = "10001:10001"

# Runner-owned environment, in the order of the reference argv (6.1).
# HF_HUB_CACHE is set explicitly (it is HF_HOME/hub anyway on v11+) so a pre-v11 image, whose ENV still
# points it at the removed read-only /hf-base, behaves the same. Offline, nothing is written there.
TMP_TMPFS_SIZE = "1g"
POLICY_ENV: tuple[tuple[str, str], ...] = (
    ("HOME", "/scratch/home"), ("HF_HOME", "/scratch/hf"), ("HF_HUB_CACHE", "/scratch/hf/hub"),
    ("TORCH_HOME", "/opt/rc/torch"),
    ("XDG_CACHE_HOME", "/scratch/xdg"), ("TRITON_CACHE_DIR", "/scratch/triton"),
    ("TORCHINDUCTOR_CACHE_DIR", "/scratch/inductor"),
    ("JAX_COMPILATION_CACHE_DIR", "/scratch/jax"), ("CUDA_CACHE_PATH", "/scratch/cuda"),
    ("NUMBA_CACHE_DIR", "/scratch/numba"),
    ("MPLCONFIGDIR", "/scratch/mpl"), ("TMPDIR", "/scratch/tmp"),
    ("PYTHONPYCACHEPREFIX", "/scratch/pycache"), ("PYTHONDONTWRITEBYTECODE", "1"),
    ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HF_DATASETS_OFFLINE", "1"),
    ("DO_NOT_TRACK", "1"), ("HF_HUB_DISABLE_TELEMETRY", "1"), ("WANDB_MODE", "disabled"),
    ("RC_WEIGHTS_DIR", "/weights"), ("RC_SUBMISSION_DIR", "/submission"), ("RC_SCRATCH", "/scratch"),
)
RESERVED_ENV_KEYS = frozenset(k for k, _ in POLICY_ENV)
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MANIFEST_DIGEST = re.compile(r"^(?P<repo>[^@\s]+)@(?P<digest>sha256:[0-9a-f]{64})$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExtraArgsRejected(ValueError):
    """A judged RC_POLICY_DOCKER_ARGS token is not on the allowlist."""


def validate_judged_extra_args(args: Sequence[str]) -> list[str]:
    """Default-deny allowlist for pinned_image mode: `-e/--env KEY=VAL` (KEY not runner-set),
    `--gpus VALUE`, and CDI `--device nvidia.com/gpu=...`. Everything else (network, dns,
    add-host, publish, mount, privileges, memory...) is rejected. Returns normalized tokens."""
    out: list[str] = []
    i = 0
    args = list(args)

    def value(opt: str, inline: str | None) -> str:
        nonlocal i
        if inline is not None:
            i += 1
            return inline
        if i + 1 >= len(args):
            raise ExtraArgsRejected(f"{opt} requires a value")
        i += 2
        return args[i - 1]

    while i < len(args):
        arg = args[i]
        opt, eq, inline = arg.partition("=")
        inline_v = inline if eq else None
        if opt in ("-e", "--env"):
            val = value(opt, inline_v)
            key, has_eq, _ = val.partition("=")
            if not has_eq or not _ENV_KEY.match(key):
                raise ExtraArgsRejected(f"{opt} {val!r}: environment overrides must be KEY=VALUE")
            if key in RESERVED_ENV_KEYS:
                raise ExtraArgsRejected(f"environment variable {key} is set by the runner and cannot be overridden")
            out += ["-e", val]
        elif opt == "--gpus":
            out += ["--gpus", value(opt, inline_v)]
        elif opt == "--device":
            val = value(opt, inline_v)
            if not val.startswith("nvidia.com/gpu="):
                raise ExtraArgsRejected(f"--device {val!r}: only CDI nvidia.com/gpu=... devices are allowed")
            out += ["--device", val]
        else:
            raise ExtraArgsRejected(f"{opt} is not allowed in judged pinned_image mode "
                                    "(network, dns, add-host, publish, mounts, privileges and memory are runner-controlled)")
        # value() advanced i
    return out


def host_platform(machine: str | None = None) -> str:
    m = (machine or _platform.machine()).lower()
    if m in ("aarch64", "arm64"):
        return "linux/arm64"
    if m in ("x86_64", "amd64"):
        return "linux/amd64"
    raise ValueError(f"unsupported host architecture {m!r}")


def parse_policy_image(ref: str) -> tuple[str, str]:
    """`repo@sha256:<manifest digest>` -> (repo, manifest digest)."""
    m = _MANIFEST_DIGEST.match(ref.strip())
    if not m:
        raise ValueError(f"RC_POLICY_IMAGE must look like repo@sha256:<64 hex>, got {ref!r}")
    return m["repo"], m["digest"]


def gpu_args(spec: str, extra: Sequence[str]) -> list[str]:
    """Runner-owned GPU flag; omitted when the allowlisted extras already carry one."""
    if any(a in ("--gpus", "--device") for a in extra) or spec.strip().lower() in ("", "none"):
        return []
    spec = spec.strip()
    if spec == "all" or spec.startswith("device="):
        return ["--gpus", spec]
    return ["--gpus", f"device={spec}"]


def scratch_mount(size: str) -> str:
    return f"/scratch:rw,nosuid,nodev,exec,size={size},mode=1777"


def policy_run_args(*, name: str, labels: Sequence[str], image_ref: str, submission_dir: str, weights_dir: str,
                    memory: str, cpus: str, pids_limit: int, user: str, shm_size: str,
                    log_max_size: str, log_max_files: int, scratch_size: str, gpus: str,
                    extra_args: Sequence[str], network: str = POLICY_NETWORK) -> list[str]:
    """The reference argv of section 6.1 (everything after `docker run -d`)."""
    extra = validate_judged_extra_args(extra_args)
    argv = ["--pull=never", "--name", name, *labels,
            "--network", network,
            "--read-only",
            "--user", user,
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--oom-score-adj", "1000",
            "--pids-limit", str(pids_limit),
            "--memory", memory, "--memory-swap", memory,
            "--cpus", cpus,
            "--ipc", "private", "--shm-size", shm_size,
            "--log-driver", "json-file", "--log-opt", f"max-size={log_max_size}", "--log-opt", f"max-file={log_max_files}",
            "--tmpfs", scratch_mount(scratch_size),
            # /tmp is a plain directory in the image (platforms may mount their own /tmp); the
            # read-only rootfs needs a writable one. TMPDIR stays /scratch/tmp. Counts against memory.
            "--tmpfs", f"/tmp:rw,nosuid,nodev,exec,size={TMP_TMPFS_SIZE},mode=1777",
            "--mount", f"type=bind,src={submission_dir},dst=/submission,readonly",
            "--mount", f"type=bind,src={weights_dir},dst=/weights,readonly"]
    argv += gpu_args(gpus, extra)
    for key, val in POLICY_ENV:
        argv += ["-e", f"{key}={val}"]
    argv += extra
    argv.append(image_ref)
    return argv


def swap_status(proc_swaps: str = "/proc/swaps") -> dict:
    """Host swap report: {"enabled": bool, "total_kib": int, "devices": [...]}. Swap lets a
    memory-capped container's host thrash; the cgroup limit is set with no swap, but operators
    should still know (docs/policy-runtime-pin.md section 8)."""
    devices: list[str] = []
    total = 0
    try:
        lines = Path(proc_swaps).read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return {"enabled": False, "total_kib": 0, "devices": [], "known": False}
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[2].isdigit():
            devices.append(parts[0])
            total += int(parts[2])
    return {"enabled": bool(devices), "total_kib": total, "devices": devices, "known": True}


def make_world_readable(tree: Path) -> None:
    """a+rX: the policy user (10001) is not the owner of the mounted /submission tree."""
    for dirpath, dirnames, filenames in os.walk(tree):
        for name in [*dirnames, *filenames]:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                continue
            mode = stat.S_IMODE(os.stat(path).st_mode)
            mode |= 0o444 | (0o111 if (os.path.isdir(path) or mode & 0o111) else 0)
            os.chmod(path, mode)
    os.chmod(tree, 0o755)


def finalize_submission_tree(tree: Path) -> list:
    """Make `tree` fit to mount as /submission: drop any team rc_sdk/rc_runtime copy (the image provides
    them; docs/policy-runtime-pin.md section 4.3) and bytecode, then a+rX. Shared by the judged worker
    and the practice launcher. Returns removed relative paths."""
    removed = [*rc_checks.strip_bytecode(tree), *rc_checks.strip_runtime_copies(tree)]
    make_world_readable(tree)
    return removed


def materialize_submission(context_dir: Path, dest: Path) -> Path:
    """Copy a prepared context to `dest` without `.git` and finalize it for mounting."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(context_dir, dest, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    finalize_submission_tree(dest)
    return dest


def readyz_state(status_code: int, payload: Any) -> str:
    """Classify a policy /readyz answer, shared by the judged warmup and practice startup:
    `load_error` (body carries `error`), `contract_mismatch` (ready but not contract v0 / schema 2),
    `ready`, or `not_ready`."""
    body = payload if isinstance(payload, dict) else {}
    if body.get("error"):
        return "load_error"
    if status_code == 200 and body.get("ready") is True:
        return "ready" if body.get("contract") == "v0" and body.get("schema_version") == 2 else "contract_mismatch"
    return "not_ready"
