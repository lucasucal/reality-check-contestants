"""Content identity for a source-built evaluator engine."""
import hashlib
from pathlib import Path


def engine_image_id(engine_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [engine_dir / name for name in ("pyproject.toml", "uv.lock", ".python-version", "docker-entrypoint.sh")]
    files += sorted(p for p in (engine_dir / "rc_engine").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        digest.update(str(path.relative_to(engine_dir)).encode() + b"\0" + path.read_bytes() + b"\0")
    return "sha256:" + digest.hexdigest()
