"""Task packs: extra tasks declared as JSON, mounted at runtime (never baked into the public image).

A pack is a directory of `*.json` files. Each file is either a pack document

    {"pack": "hidden-v1", "tasks": [
        {"task_id": "cup_place_far_v0", "kind": "place", "visibility": "hidden",
         "description": "...", "params": {"cup_r": [0.25, 0.30], "episode_length": 300}}]}

or one task object on its own (`{"task_id": ..., "kind": "tabletop", ...}`), which
is how the fastener tasks keep one file per task (`engine/packs-hidden/`).

`kind` selects the task family. Cup tasks override CupParams fields in `params`.
Tabletop tasks are the multi-part fastener schema validated by
`tasks.tabletop.TableTask.from_json` (parts, goals, decoys, order); a hidden one
needs wire_id '1' to '4', unique across the mounted packs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .tasks import Catalog


class PackError(ValueError):
    pass


def load_packs(directory: str | Path, catalog: Catalog) -> list[str]:
    """Register every task in `directory`'s pack files; returns the new task ids (sorted)."""
    from .tasks.cups import CupParams, CupsEnv, LayoutError, check_layouts
    from .tasks.placeholder import PlaceholderEnv, hidden_spec
    from .tasks.tabletop import HIDDEN_WIRE_IDS, TableTask, TabletopEnv

    from .tasks.variation import load_eval_profile

    root = Path(directory)
    if not root.is_dir():
        raise PackError(f"task pack directory {root} does not exist")
    try:
        # The organizer-private eval profile rides with the hidden packs (profiles/eval.json).
        load_eval_profile(root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise PackError(f"profiles/eval.json: {exc}") from exc
    added: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            doc = json.loads(path.read_text())
            tasks = [doc] if isinstance(doc, dict) and "task_id" in doc else doc["tasks"]
            if not isinstance(tasks, list):
                raise ValueError("'tasks' must be a list")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise PackError(f"{path.name}: unreadable pack ({exc!r})") from exc
        for t in tasks:
            try:
                task_id, kind = t["task_id"], t["kind"]
                visibility = t.get("visibility", "hidden")
                if visibility not in ("public", "hidden"):
                    raise ValueError(f"visibility {visibility!r}")
                if kind == "placeholder":
                    wire_id = str(t.get("wire_id") or "")
                    if visibility == "hidden" and wire_id not in HIDDEN_WIRE_IDS:
                        raise ValueError(f"hidden placeholder wire_id must be one of {HIDDEN_WIRE_IDS}, got {wire_id!r}")
                    spec = hidden_spec(
                        task_id,
                        instruction=str(t.get("instruction") or ""),
                        description=str(t.get("description") or ""),
                        wire_id=wire_id,
                        visibility=visibility,
                    )
                    catalog.add(task_id, lambda s=spec: PlaceholderEnv(s),
                                wire_id=wire_id if visibility == "hidden" else "")
                elif kind == "tabletop":
                    table_task = TableTask.from_pack(t, path.parent)
                    catalog.add(task_id, lambda s=table_task: TabletopEnv(s),
                                wire_id=table_task.wire_id if table_task.visibility == "hidden" else "")
                else:
                    params = CupParams.from_json(kind, {"description": t.get("description", ""), **t.get("params", {})})
                    check_layouts(params)
                    catalog.add(task_id, lambda i=task_id, p=params, v=visibility: CupsEnv(i, p, v))
            except (KeyError, ValueError, TypeError, LayoutError) as exc:
                raise PackError(f"{path.name}: bad task {t.get('task_id', '?')!r}: {exc}") from exc
            added.append(task_id)
    return sorted(added)
