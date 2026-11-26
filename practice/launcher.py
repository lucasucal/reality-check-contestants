"""Local practice launch of the published policy image.

The policy container is started through `DockerDriver.run_policy_pinned`, the same method the
judged worker calls, so the argv (mounts, read-only /submission and /weights, offline env,
hardening, curated cache mount) is the judged argv by construction. Only the inputs differ
(paths, container name, GPU choice) and they are printed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO

from rc_runner import launch
from rc_runner.docker_driver import DockerDriver, DockerError, ImageMissing, subprocess_runner
from rc_runner.policy_limits import PolicyLimits

PRACTICE_SUBMISSION_ID = 0
PULL_TIMEOUT_SECONDS = 3600


class LaunchError(Exception):
    pass


@dataclass(frozen=True)
class PolicySettings:
    """The runner's judged policy limits (RC_POLICY_* defaults), overridable only where a laptop needs it."""
    memory: str
    cpus: str
    pids_limit: int
    user: str
    shm_size: str
    log_max_size: str
    log_max_files: int
    scratch_size: str
    gpus: str
    extra_args: tuple = ()

    @classmethod
    def from_config(cls, cfg: PolicyLimits, *, gpus: Optional[str] = None, memory: Optional[str] = None,
                    cpus: Optional[str] = None) -> "PolicySettings":
        return cls(memory=memory or cfg.policy_memory, cpus=cpus or cfg.policy_cpus, pids_limit=cfg.policy_pids_limit,
                   user=cfg.policy_user, shm_size=cfg.policy_shm_size, log_max_size=cfg.policy_log_max_size,
                   log_max_files=cfg.policy_log_max_files, scratch_size=cfg.policy_scratch_size,
                   gpus=gpus if gpus is not None else cfg.policy_gpus, extra_args=tuple(cfg.split_policy_args()))


@dataclass(frozen=True)
class ImageRef:
    repository: str
    manifest_digest: str
    platform: str
    platform_digest: str

    @property
    def run_ref(self) -> str:
        return f"{self.repository}@{self.platform_digest}"

    @property
    def pull_ref(self) -> str:
        return f"{self.repository}@{self.manifest_digest}"


RELEASE_HELP = ("The practice tool takes the policy image from the published release manifest: the participant bundle "
                "ships it as runtime/release.json (pass another file with --release FILE or RC_RELEASE_JSON). Use the "
                "release.json organizers publish with the image; the copy inside the image has no digests.")


def image_ref_from_release(release: Any, platform: Optional[str] = None, source: str = "release.json") -> ImageRef:
    """Published image for this host (or `platform`), selected exactly as the runner does."""
    image = (release.raw.get("image") or {}) if hasattr(release, "raw") else {}
    try:
        repo, manifest = launch.parse_policy_image(f"{image.get('repository', '')}@{image.get('manifest_digest', '')}")
    except ValueError:
        raise LaunchError(f"{source} does not name the policy image (image.repository and image.manifest_digest "
                          f"sha256:<64 hex> are required). {RELEASE_HELP}") from None
    plat = platform or launch.host_platform()
    digest = (image.get("platform_digests") or {}).get(plat)
    if not isinstance(digest, str) or not launch.DIGEST_RE.match(digest):
        raise LaunchError(f"{source} has no {plat} digest in image.platform_digests. {RELEASE_HELP}")
    return ImageRef(repo, manifest, plat, digest)


def policy_launch_args(driver: DockerDriver, *, name: str, image_ref: str, submission_dir: str, weights_dir: str,
                       base_cache_dir: str, settings: PolicySettings, submission_id: int = PRACTICE_SUBMISSION_ID) -> list:
    """The exact `docker run` arguments (after `docker run -d`) for these inputs: the same function
    the driver uses, so tests and the CLI's --print-argv show what will run."""
    return launch.policy_run_args(
        name=name, labels=driver._labels(submission_id), image_ref=image_ref, submission_dir=submission_dir,
        weights_dir=weights_dir, base_cache_dir=base_cache_dir, memory=settings.memory, cpus=settings.cpus,
        pids_limit=settings.pids_limit, user=settings.user, shm_size=settings.shm_size,
        log_max_size=settings.log_max_size, log_max_files=settings.log_max_files, scratch_size=settings.scratch_size,
        gpus=settings.gpus, extra_args=settings.extra_args)


def start_policy(driver: DockerDriver, *, name: str, image_ref: str, submission_dir: str, weights_dir: str,
                 base_cache_dir: str, settings: PolicySettings, submission_id: int = PRACTICE_SUBMISSION_ID) -> str:
    """Start the policy container via the judged code path; returns its internal-network IP."""
    driver.remove(name)
    try:
        driver.run_policy_pinned(
            name=name, image_ref=image_ref, submission_id=submission_id, submission_dir=submission_dir,
            weights_dir=weights_dir, base_cache_dir=base_cache_dir, memory=settings.memory, cpus=settings.cpus,
            pids_limit=settings.pids_limit, user=settings.user, shm_size=settings.shm_size,
            log_max_size=settings.log_max_size, log_max_files=settings.log_max_files,
            scratch_size=settings.scratch_size, gpus=settings.gpus, extra_args=settings.extra_args)
        # The pinned launch attaches the policy to launch.POLICY_NETWORK, not the legacy driver default (CS-B-2).
        return driver.container_ip(name, launch.POLICY_NETWORK)
    except DockerError as exc:
        raise LaunchError(f"could not start the policy container: {exc}") from exc


def ensure_image(driver: DockerDriver, image: ImageRef, *, pull: bool = True, out: Optional[TextIO] = None) -> dict:
    """The platform image must be present locally (judged launches use --pull=never). Pulling by the
    manifest digest happens here, before launch, where network access is allowed."""
    out = out or sys.stderr

    def fetch(ref: str) -> None:
        print(f"pulling {ref} ...", file=out, flush=True)
        result = subprocess_runner(["docker", "pull", ref], PULL_TIMEOUT_SECONDS, None)
        if result.returncode != 0:
            raise LaunchError(f"docker pull {ref} failed: {result.output.strip()[-500:]}")

    try:
        info = driver.image_inspect(image.run_ref)
    except ImageMissing:
        if not pull:
            raise LaunchError(f"image {image.run_ref} is not present locally (re-run without --no-pull to fetch it)") from None
        fetch(image.pull_ref)
        try:
            info = driver.image_inspect(image.run_ref)
        except ImageMissing:
            # Docker's containerd image store keeps an image pulled by its manifest-list digest under that digest
            # only, so the platform reference the launch uses does not resolve yet. Pulling the platform digest
            # itself reuses the layers just downloaded and makes it resolvable (classic store: never reached).
            fetch(image.run_ref)
            info = driver.image_inspect(image.run_ref)
    arch = info.get("Architecture")
    if arch and arch != image.platform.split("/", 1)[1]:
        raise LaunchError(f"image {image.run_ref} is {arch}, expected {image.platform}")
    return info


def gpu_note(settings: PolicySettings) -> str:
    if launch.gpu_args(settings.gpus, settings.extra_args):
        return f"GPU: {settings.gpus}"
    return "GPU: none requested (a GPU-dependent policy may fail on this host; pass --gpus all)"
