"""One next step for the setup failures teams hit most (Docker down, no disk, image not pulled, engine not
reachable). Matching is on the error text practice already prints; nothing here changes what runs."""
from __future__ import annotations

import re
from typing import Optional

from rc_runner.practice import ENGINE_NEXT_STEP

_DOCKER_DOWN = re.compile(r"cannot connect to the docker daemon|is the docker daemon running|docker daemon is not "
                          r"running|error during connect|cannot execute 'docker'|no such file or directory: 'docker'",
                          re.I)
_DISK = re.compile(r"no space left on device|not enough space|disk quota exceeded", re.I)
_PULL = re.compile(r"docker pull .* failed|manifest unknown|pull access denied|toomanyrequests|"
                   r"tls handshake timeout|i/o timeout", re.I)
_NOT_PULLED = re.compile(r"is not present locally", re.I)
_ENGINE = re.compile(r"engine (GET|POST) \S+ failed|engine not ready", re.I)
_GPU = re.compile(r"could not select device driver|nvidia-container|unknown or invalid runtime name: nvidia", re.I)

DOCKER_DOWN = "start Docker (Docker Desktop, or `sudo systemctl start docker`), check `docker info` works, then re-run."
DISK = ("free disk space: the policy image needs about 20 GB; `docker system df` shows usage and "
        "`docker image prune` removes unused images.")
PULL = "check your network and that the release manifest is the one organizers published, then re-run (the pull resumes)."
NOT_PULLED = "re-run without --no-pull so practice can fetch the image once."
GPU = "this machine has no usable NVIDIA GPU for Docker: re-run with --gpus none."


def next_step(text: str, engine_url: str = "http://127.0.0.1:8090") -> Optional[str]:
    """The single most likely fix for an error message, or None when there is no confident match."""
    if "Next step:" in text:  # the message already says it
        return None
    if _DOCKER_DOWN.search(text):
        return DOCKER_DOWN
    if _DISK.search(text):
        return DISK
    if _GPU.search(text):
        return GPU
    if _NOT_PULLED.search(text):
        return NOT_PULLED
    if _PULL.search(text):
        return PULL
    if _ENGINE.search(text):
        return ENGINE_NEXT_STEP.format(url=engine_url.rstrip("/")) + "."
    return None
