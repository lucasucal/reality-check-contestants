"""Policy container limits from RC_* environment variables, with the judged defaults.

The runner's Config and the practice launcher both read them here, so practice launches the policy with
the same limits as a judged run.
"""
from __future__ import annotations

import os
import shlex

from . import launch


def int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def str_env(name: str, default: str) -> str:
    return os.environ.get(name, "") or default


class PolicyLimits:
    """Docker binary and policy container limits."""

    def __init__(self) -> None:
        self.docker_bin: str = str_env("RC_DOCKER_BIN", "docker")
        # Team container limits + extra args (e.g. "--gpus all").
        self.policy_memory: str = str_env("RC_POLICY_MEMORY", "100g")
        self.policy_cpus: str = str_env("RC_POLICY_CPUS", "8")
        self.policy_pids_limit: int = int_env("RC_POLICY_PIDS_LIMIT", 1024)
        self.policy_docker_args: str = os.environ.get("RC_POLICY_DOCKER_ARGS", "")
        self.policy_scratch_size: str = str_env("RC_POLICY_SCRATCH_SIZE", "8g")
        self.policy_shm_size: str = str_env("RC_POLICY_SHM_SIZE", "1g")
        self.policy_log_max_size: str = str_env("RC_POLICY_LOG_MAX_SIZE", "50m")
        self.policy_log_max_files: int = int_env("RC_POLICY_LOG_MAX_FILES", 3)
        self.policy_gpus: str = str_env("RC_POLICY_GPUS", "all")  # all | none | device list
        self.policy_user: str = str_env("RC_POLICY_USER", "10001:10001")
        self.policy_port: int = launch.POLICY_PORT  # fixed by the image contract; not an env knob

    def split_policy_args(self) -> list[str]:
        return shlex.split(self.policy_docker_args)
