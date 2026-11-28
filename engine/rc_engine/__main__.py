"""Entry point: python -m rc_engine --help."""
from __future__ import annotations

import argparse
import os
import signal

from .embodiment import DEFAULT as DEFAULT_EMBODIMENT, REGISTERED, resolve
from .packs import PackError, load_packs
from .server import EngineServer
from .tasks import builtin_catalog


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rc_engine")
    parser.add_argument("--host", default=os.environ.get("RC_ENGINE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RC_ENGINE_PORT", "8090")))
    parser.add_argument("--tasks", default="", help="comma-separated task ids to serve (default: every task in the catalog)")
    parser.add_argument("--task-packs", default="", help="directory of JSON task packs to add (e.g. hidden tasks)")
    parser.add_argument("--jpeg-quality", type=int, default=90, choices=range(1, 96), metavar="1..95")
    parser.add_argument("--embodiment", default=os.environ.get("RC_EMBODIMENT", DEFAULT_EMBODIMENT),
                        help=f"cell id (registered: {', '.join(sorted(REGISTERED))})")
    parser.add_argument("--audience-cameras", action="store_true",
                        default=os.environ.get("RC_AUDIENCE_CAMERAS") == "1",
                        help="render extra booth/side/orbit views for organizer live debug (or RC_AUDIENCE_CAMERAS=1)")
    args = parser.parse_args(argv)
    try:
        resolve(args.embodiment)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        from .arm_limits import safety_stop_mode  # noqa: PLC0415 - MuJoCo-free engines skip the check
    except ImportError:
        pass
    else:
        try:
            mode = safety_stop_mode()
        except ValueError as exc:
            parser.error(str(exc))
        if mode == "enforce":
            print("[rc_engine] protective stop enforced (RC_SAFETY_STOP=enforce)", flush=True)

    catalog = builtin_catalog()
    if args.task_packs:
        try:
            catalog_new = load_packs(args.task_packs, catalog)
        except PackError as exc:
            parser.error(str(exc))
        print(f"[rc_engine] loaded {len(catalog_new)} pack tasks", flush=True)  # ids stay out of the log: hidden
    # Leave extra booth/side/orbit cameras off unless organizer live-debug opted in,
    # so scoring and practice do not pay 2x shadow-mapped rendering + JPEG encoding inside scenario time.
    if args.audience_cameras:
        os.environ["RC_AUDIENCE_CAMERAS"] = "1"
    else:
        os.environ.pop("RC_AUDIENCE_CAMERAS", None)
    task_ids = [t for t in args.tasks.split(",") if t] or list(catalog.defaults)
    unknown = sorted(set(task_ids) - set(catalog.factories))
    if unknown:
        parser.error(f"unknown tasks: {', '.join(unknown)}")
    server = EngineServer((args.host, args.port), task_ids, args.jpeg_quality, factory=catalog.make)
    print(f"[rc_engine] serving {len(task_ids)} tasks on {args.host}:{args.port}", flush=True)
    # `docker stop` sends SIGTERM, which PID 1 otherwise ignores: route it through the same clean shutdown as
    # Ctrl+C so server_close() frees the renderers (A7) instead of the container being killed at the timeout.
    def _terminate(signum: int, frame: object) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
