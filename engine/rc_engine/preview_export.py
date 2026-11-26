"""Pre-render the public task preview to static files.

The GCP VM runs no simulation, so the dashboard's task preview is a set of JPEGs
rendered here once and served as static files. Each public task gets the same
pictures the live preview showed on seed 0: the policy cameras, the goal image,
the two 45° spectator views, and the orbit camera at fixed azimuths.

    python -m rc_engine.preview_export ../platform/public/tasks

Re-run it after changing a public scene, its cameras or its goal image.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Any

from .preview import _make_envs, _packet, _views
from .schema import encode_obs
from .tasks import builtin_catalog
from .tasks.tabletop import PUBLIC_IDS

ORBIT_ELEVATION = 32.0
ORBIT_STEP = 15
ORBIT_AZIMUTHS = tuple(range(0, 360, ORBIT_STEP))
JPEG_QUALITY = 90
MANIFEST = "manifest.json"


def _write(root: Path, relative: str, b64: str) -> str:
    (root / relative).write_bytes(base64.b64decode(b64))
    return relative


def export(out: Path, task_ids: tuple[str, ...] = PUBLIC_IDS) -> dict[str, Any]:
    """Render every public task into out/<task>/ and write out/manifest.json. Returns the manifest."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    catalog = builtin_catalog()
    tasks = []
    for scene_id in task_ids:
        if scene_id not in PUBLIC_IDS:
            raise ValueError(f"{scene_id} is not a public task; only public scenes may be exported")
        (out / scene_id).mkdir()
        env = _make_envs(catalog, (scene_id,))[scene_id]
        try:
            env.set_orbit(ORBIT_AZIMUTHS[0], ORBIT_ELEVATION)  # type: ignore[attr-defined]
            obs = encode_obs(env.spec, 0, env.reset(0), JPEG_QUALITY)
            packet = _packet(obs, "preview")
            views = _views(env) or {}
            cameras = {name: _write(out, f"{scene_id}/camera-{name}.jpg", b64)
                       for name, b64 in packet.get("images", {}).items()}
            orbit = []
            for azimuth in ORBIT_AZIMUTHS:
                env.set_orbit(azimuth, ORBIT_ELEVATION)  # type: ignore[attr-defined]
                env._obs()  # noqa: SLF001 - re-render the same state from the new orbit angle
                orbit.append(_write(out, f"{scene_id}/orbit-{azimuth:03d}.jpg", (_views(env) or {})["orbit"]))
            proprio = packet.get("proprio") or {}
            tasks.append({
                "id": scene_id,
                "task_id": packet.get("task_id"),
                "instruction": packet.get("instruction"),
                "episode_length": env.spec.episode_length,
                "scene": next(iter(cameras.values()), None),
                "goal": _write(out, f"{scene_id}/goal.jpg", obs["goal"]),
                "views": {name: _write(out, f"{scene_id}/{name}.jpg", views[name])
                          for name in ("side", "booth") if name in views},
                "orbit": orbit,
                "cameras": cameras,
                "proprio": {key: proprio.get(key) for key in ("joint_pos", "joint_vel", "gripper")},
            })
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
    manifest = {
        "seed": 0,
        "orbit": {"elevation": ORBIT_ELEVATION, "azimuths": list(ORBIT_AZIMUTHS)},
        "tasks": tasks,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rc_engine.preview_export")
    parser.add_argument("out", type=Path, help="output directory; replaced if it exists")
    args = parser.parse_args(argv)
    manifest = export(args.out)
    print(f"[rc_preview_export] wrote {len(manifest['tasks'])} tasks to {args.out}", flush=True)


if __name__ == "__main__":
    main()
