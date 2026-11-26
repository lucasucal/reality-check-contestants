"""Docker driver: CLI subprocess (plan §7 rationale), with an injectable command
runner so the whole lifecycle is unit-testable without a daemon.

No host networking, ever, and never the shared default bridge: engine and policy
containers sit on separate ICC-disabled bridge networks. Only the engine publishes a loopback port.
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import launch

LABEL = "rc.runner=1"
ENGINE_CONTAINER_PORT = 8090
ENGINE_NETWORK = "rc-engine-net"


def _reject_memory_overrides(args: Sequence[str]) -> None:
    """Keep extra Docker args from silently replacing the runner's memory ceiling."""
    forbidden = {"-m", "--memory", "--memory-swap"}
    for arg in args:
        option = arg.split("=", 1)[0]
        if option in forbidden or arg.startswith("-m") and not arg.startswith("--"):
            raise DockerError(f"{arg.split('=', 1)[0]} is controlled by the runner memory limit")


@dataclass
class CmdResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


class CommandTimeout(Exception):
    """A docker command exceeded its time budget (output so far in `.output`)."""

    def __init__(self, seconds: float, output: str = "") -> None:
        super().__init__(f"timed out after {seconds:g}s")
        self.output = output


class CommandCancelled(Exception):
    """cancel() fired while a docker command ran (shutdown or lease lost)."""

    def __init__(self, output: str = "") -> None:
        super().__init__("command cancelled")
        self.output = output


class DockerError(Exception):
    """A docker command that must succeed did not (daemon down, bad flags, GPU...)."""


# (argv, timeout seconds | None, cancel() -> bool | None) -> CmdResult
CommandRunner = Callable[[Sequence[str], "float | None", "Callable[[], bool] | None"], CmdResult]
# (runner, docker_bin, network, cancel() -> bool | None) -> None; raises DockerError when the host is not ready
PolicyPreflight = Callable[[CommandRunner, str, str, "Callable[[], bool] | None"], None]


def subprocess_runner(argv: Sequence[str], timeout: float | None = None,
                      cancel: Callable[[], bool] | None = None) -> CmdResult:
    """Default command runner: Popen + 0.2 s polling so cancel/timeout can kill a
    slow Docker command instead of blocking for its whole budget."""
    try:
        proc = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                errors="replace")
    except OSError as exc:
        raise DockerError(f"cannot execute {argv[0]!r}: {exc}") from exc
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.2)
            return CmdResult(proc.returncode, out or "", err or "")
        except subprocess.TimeoutExpired:
            pass
        if cancel is not None and cancel():
            out, err = _kill(proc)
            raise CommandCancelled((out or "") + (err or ""))
        if deadline is not None and time.monotonic() > deadline:
            out, err = _kill(proc)
            raise CommandTimeout(timeout or 0, (out or "") + (err or ""))


def _kill(proc: "subprocess.Popen[str]") -> tuple[str, str]:
    proc.kill()
    try:
        return proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return "", ""


def free_port() -> int:
    """An ephemeral loopback port (small TOCTOU window, acceptable on one box)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ImageMissing(DockerError):
    """`docker image inspect` could not find the pinned image locally (never pulled: --pull=never)."""


def _network_arg(extra_args: Sequence[str]) -> str | None:
    """Return an explicit --network/--net value from extra_args, rejecting 'host'."""
    chosen: str | None = None
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        val: str | None = None
        if arg in ("--network", "--net"):
            if i + 1 >= len(extra_args):
                raise DockerError(f"{arg} requires a network name")
            val = extra_args[i + 1]
            i += 2
        elif arg.startswith("--network="):
            val = arg.split("=", 1)[1]
            i += 1
        elif arg.startswith("--net="):
            val = arg.split("=", 1)[1]
            i += 1
        else:
            i += 1
            continue
        if val.strip() == "host":
            raise DockerError("host networking (--network host) is forbidden")
        chosen = val.strip()
    return chosen


class DockerDriver:
    def __init__(self, docker_bin: str = "docker", runner: CommandRunner | None = None,
                 attempt_id: str = "", *, node_id: str | None = None,
                 policy_preflight: PolicyPreflight | None = None) -> None:
        self._bin = docker_bin
        self._run = runner or subprocess_runner
        # Host check run before every pinned policy launch: (runner, docker_bin, network, cancel), raises DockerError.
        # The judged worker passes its own (worker.default_deps); a team's own machine has none.
        self._policy_preflight = policy_preflight
        self._node_id = node_id
        self._attempt_id = attempt_id
        self._ensured_networks: set[str] = set()

    def _labels(self, submission_id: int) -> list[str]:
        labels = ["--label", LABEL]
        if self._node_id:
            labels += ["--label", f"rc.node_id={self._node_id}"]
        labels += ["--label", f"rc.submission_id={submission_id}"]
        if self._attempt_id:
            labels += ["--label", f"rc.attempt_id={self._attempt_id}"]
        return labels

    def ensure_network(self, name: str, cancel: Callable[[], bool] | None = None) -> None:
        """Idempotently create an isolated bridge network with ICC disabled."""
        if name in self._ensured_networks:
            return
        inspect_cmd = [self._bin, "network", "inspect", name]
        if self._run(inspect_cmd, 30, cancel).returncode == 0:
            self._ensured_networks.add(name)
            return
        create = self._run([
            self._bin, "network", "create", "--driver", "bridge",
            "--label", LABEL,
            "-o", "com.docker.network.bridge.enable_icc=false",
            name,
        ], 30, cancel)
        if create.returncode != 0 and self._run(inspect_cmd, 30, cancel).returncode != 0:
            raise DockerError(f"docker network create {name} failed ({create.returncode}): {create.output.strip()[-500:]}")
        self._ensured_networks.add(name)

    def ensure_internal_network(self, name: str = launch.POLICY_NETWORK,
                                cancel: Callable[[], bool] | None = None) -> None:
        """Idempotently create the internal (no egress, no gateway) ICC-disabled policy bridge
        (docs/policy-runtime-pin.md section 6). A non-internal network of the same name
        is refused rather than silently reused: it would give the policy an outbound route. Pinned
        mode uses its own name (launch.POLICY_NETWORK), so this only trips on manual tampering."""
        key = f"internal:{name}"
        if key in self._ensured_networks:
            return
        inspect_cmd = [self._bin, "network", "inspect", "-f", "{{.Internal}}", name]
        existing = self._run(inspect_cmd, 30, cancel)
        if existing.returncode == 0:
            if existing.stdout.strip().lower() == "false":
                raise DockerError(f"docker network {name} exists but is not --internal; "
                                  f"remove it with `docker network rm {name}` while no policy containers run")
            self._ensured_networks.add(key)
            return
        create = self._run([self._bin, "network", "create", "--internal", "--driver", "bridge",
                            "-o", "com.docker.network.bridge.enable_icc=false",
                            "--label", LABEL, name], 30, cancel)
        if create.returncode != 0 and self._run(inspect_cmd, 30, cancel).returncode != 0:
            raise DockerError(f"docker network create --internal {name} failed ({create.returncode}): "
                              f"{create.output.strip()[-500:]}")
        self._ensured_networks.add(key)

    # -- run / lifecycle ------------------------------------------------------
    def run_detached(self, args: Sequence[str], cancel: Callable[[], bool] | None = None) -> str:
        """`docker run -d <args>`; returns the container id. Raises DockerError."""
        result = self._run([self._bin, "run", "-d", *args], 120, cancel)
        if result.returncode != 0:
            raise DockerError(f"docker run failed ({result.returncode}): {result.output.strip()[-1500:]}")
        return result.stdout.strip()

    def run_policy_pinned(self, *, name: str, image_ref: str, submission_id: int, submission_dir: str,
                          weights_dir: str, base_cache_dir: str, memory: str, cpus: str, pids_limit: int,
                          user: str, shm_size: str, log_max_size: str, log_max_files: int, scratch_size: str,
                          gpus: str, extra_args: Sequence[str],
                          cancel: Callable[[], bool] | None = None) -> str:
        """Start the pinned policy image (contract 6.1). No build, no `-p`, `--pull=never`."""
        _reject_memory_overrides(extra_args)
        try:
            argv = launch.policy_run_args(
                name=name, labels=self._labels(submission_id), image_ref=image_ref,
                submission_dir=submission_dir, weights_dir=weights_dir, base_cache_dir=base_cache_dir,
                memory=memory, cpus=cpus, pids_limit=pids_limit, user=user, shm_size=shm_size,
                log_max_size=log_max_size, log_max_files=log_max_files, scratch_size=scratch_size,
                gpus=gpus, extra_args=extra_args)
        except launch.ExtraArgsRejected as exc:
            raise DockerError(str(exc)) from exc
        self.ensure_internal_network(launch.POLICY_NETWORK, cancel)
        if self._policy_preflight is not None:
            self._policy_preflight(self._run, self._bin, launch.POLICY_NETWORK, cancel)
        return self.run_detached(argv, cancel)

    def confirms_scratch_failure(self, *, name: str, image_ref: str, submission_id: int,
                                 user: str, memory: str, scratch_size: str,
                                 cancel: Callable[[], bool] | None = None) -> bool:
        """Repeat scratch initialization without mounting or running any team code."""
        probe = """from pathlib import Path
from rc_runtime.loader import RuntimeStartupError, create_scratch

try:
    create_scratch(Path("/scratch"))
except RuntimeStartupError as exc:
    if exc.code != "scratch_unwritable":
        raise
    print("rc-scratch-unwritable", flush=True)
    raise SystemExit(74)
"""
        try:
            result = self._run([
                self._bin, "run", "--rm", "--pull=never", "--name", name, *self._labels(submission_id),
                "--network", "none", "--read-only", "--user", user,
                "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
                "--memory", memory, "--memory-swap", memory, "--pids-limit", "64",
                "--tmpfs", launch.scratch_mount(scratch_size),
                "--entrypoint", "python", image_ref, "-I", "-B", "-c", probe,
            ], 10, cancel)
        except (CommandTimeout, CommandCancelled, DockerError):
            self.remove(name)
            return False
        return result.returncode == 74 and result.stdout.strip() == "rc-scratch-unwritable"

    def image_inspect(self, ref: str) -> dict[str, Any]:
        """`docker image inspect <ref>` -> first object. ImageMissing when absent."""
        result = self._run([self._bin, "image", "inspect", ref], 60, None)
        if result.returncode != 0:
            raise ImageMissing(f"image {ref} not present on this host: {result.output.strip()[-300:]}")
        try:
            data = json.loads(result.stdout)
            return data[0] if isinstance(data, list) else data
        except (ValueError, IndexError, TypeError) as exc:
            raise DockerError(f"unparseable docker image inspect output for {ref}") from exc

    def container_ip(self, name: str, network: str = launch.POLICY_NETWORK) -> str:
        """The container's IPv4 address on `network` (the runner reaches the policy by it)."""
        fmt = '{{(index .NetworkSettings.Networks "%s").IPAddress}}' % network
        result = self._run([self._bin, "inspect", "-f", fmt, name], 30, None)
        ip = result.stdout.strip()
        if result.returncode != 0 or not re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip):
            raise DockerError(f"cannot determine {name}'s address on {network}: {result.output.strip()[-300:]}")
        return ip

    def exit_code(self, name: str) -> int | None:
        result = self._run([self._bin, "inspect", "-f", "{{.State.ExitCode}}", name], 30, None)
        try:
            return int(result.stdout.strip()) if result.returncode == 0 else None
        except ValueError:
            return None

    def oom_killed(self, name: str) -> bool:
        """True when docker reports the container was killed by the cgroup OOM killer."""
        result = self._run([self._bin, "inspect", "-f", "{{.State.OOMKilled}}", name], 30, None)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def inspect_run_image(self, name: str, *, expected_manifest_digest: str | None = None) -> dict[str, Any]:
        """Provenance of the image a started container really runs, read from docker (never from
        config): platform digest from the reference the container was created with, image id and
        platform from the image object, manifest digest only when RepoDigests confirms it."""
        result = self._run([self._bin, "inspect", "-f", "{{.Image}} {{.Config.Image}}", name], 30, None)
        parts = result.stdout.split()
        if result.returncode != 0 or len(parts) != 2:
            raise DockerError(f"cannot inspect the image of container {name}: {result.output.strip()[-300:]}")
        image_id, config_ref = parts
        image = self.image_inspect(image_id)
        info: dict[str, Any] = {"image_id": image.get("Id") or image_id,
                                "platform": f"{image.get('Os', 'linux')}/{image.get('Architecture', '')}"}
        if "@" in config_ref and launch.DIGEST_RE.match(config_ref.split("@", 1)[1]):
            info["platform_digest"] = config_ref.split("@", 1)[1]
        repo_digests = [d for d in (image.get("RepoDigests") or []) if isinstance(d, str)]
        if repo_digests:
            info["repo_digests"] = repo_digests
        if expected_manifest_digest:
            for d in repo_digests:
                if d.endswith("@" + expected_manifest_digest):
                    info["manifest_digest"] = expected_manifest_digest
        return info

    def run_engine(self, *, name: str, image: str, submission_id: int, host_port: int, extra_args: Sequence[str],
                   packs_dir: str | None, jpeg_quality: int, tasks: Sequence[str] | None = None,
                   embodiment: str = "ur12e_robotiq_2f140", cancel: Callable[[], bool] | None = None,
                   memory: str = "4g", local_only: bool = False) -> str:
        _reject_memory_overrides(extra_args)
        if local_only and any(arg == "--pull" or arg.startswith("--pull=") for arg in extra_args):
            raise DockerError("--pull is controlled by the judged engine launch")
        net_args: list[str] = []
        if _network_arg(extra_args) is None:
            self.ensure_network(ENGINE_NETWORK, cancel)
            net_args = ["--network", ENGINE_NETWORK]
        args = ["--name", name, *self._labels(submission_id),
                *net_args,
                "--memory", memory, "--memory-swap", memory,
                "-p", f"127.0.0.1:{host_port}:{ENGINE_CONTAINER_PORT}", *extra_args,
                "-e", f"RC_EMBODIMENT={embodiment}"]
        if local_only:
            args.insert(0, "--pull=never")
        if packs_dir:
            args += ["-v", f"{packs_dir}:/packs:ro"]
        args.append(image)
        if packs_dir:
            args += ["--task-packs", "/packs"]
        if tasks:
            args += ["--tasks", ",".join(tasks)]
        args += ["--jpeg-quality", str(jpeg_quality)]
        return self.run_detached(args, cancel)

    def kill(self, name: str) -> None:
        """`docker kill` (SIGKILL); a container that is already gone is fine."""
        try:
            self._run([self._bin, "kill", name], 30, None)
        except (CommandTimeout, DockerError, CommandCancelled):
            pass

    def is_running(self, name: str) -> bool:
        result = self._run([self._bin, "inspect", "-f", "{{.State.Running}}", name], 30, None)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def logs_tail(self, name: str, lines: int = 40) -> str:
        try:
            result = self._run([self._bin, "logs", "--tail", str(lines), name], 30, None)
        except (CommandTimeout, DockerError):
            return ""
        return result.output.strip()

    def logs_all(self, name: str) -> str:
        result = self._run([self._bin, 'logs', name], 30, None)
        if result.returncode != 0:
            raise DockerError(f'could not collect container logs for {name}: {result.output[-500:]}')
        return result.output

    def remove(self, name: str) -> None:
        """Best-effort `docker rm -f` (a missing container is fine)."""
        try:
            self._run([self._bin, "rm", "-f", name], 60, None)
        except (CommandTimeout, DockerError, CommandCancelled):
            pass

    # -- orphan reaping (plan §3.4) -------------------------------------------
    def reap_orphans(self) -> list[str]:
        """Remove orphaned containers owned by this runner worker (scoped to rc.node_id
        when set so multi-slot Spark hosts never kill sibling workers' live containers)."""
        filters = ["--filter", f"label={LABEL}"]
        if self._node_id:
            filters += ["--filter", f"label=rc.node_id={self._node_id}"]
        result = self._run([self._bin, "ps", "-a", *filters, "--format", "{{.Names}}"], 60, None)
        if result.returncode != 0:
            raise DockerError(f"docker ps failed: {result.output.strip()[-500:]}")
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for name in names:
            self.remove(name)
        return names
