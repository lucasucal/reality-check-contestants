"""Reality Check runner: runs a policy against the engine's tasks (judged runs and
`python -m rc_runner.practice`)."""
from __future__ import annotations


def log(message: str) -> None:
    """One log line, engine-style prefix (engine/rc_engine/__main__.py convention)."""
    print(f"[rc_runner] {message}", flush=True)
