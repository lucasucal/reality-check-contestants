"""CLI: python -m rc_checks {repo|snapshot|dtypes} <dir> [--release path] [--json]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import check_dtypes, check_repo, check_snapshot, load_release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rc_checks")
    parser.add_argument("command", choices=["repo", "snapshot", "dtypes"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--release", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        print(f"not a directory: {args.directory}", file=sys.stderr)
        return 2
    release = load_release(args.release) if args.command != "snapshot" or args.release else None
    if args.command == "repo":
        result = check_repo(args.directory, release=release)
    elif args.command == "snapshot":
        result = check_snapshot(args.directory, release=release)
    else:
        result = check_dtypes(args.directory, release=release)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        for finding in result.to_dict()["findings"]:
            print(f"{finding['severity']} {finding['code']} {finding['path'] or '-'}: {finding['message']}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
