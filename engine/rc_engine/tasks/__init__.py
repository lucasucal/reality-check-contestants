"""Task catalog: task_id -> env factory.

The contest default is the public fastener tasks on the printed fixture (two: bolt_to_tray, bolt_and_nut_to_fixture).
Placeholders, bench lifts, and cup tasks stay constructable and are not served.
Hidden tasks are not in this catalog; a pack mounts them on the eval host.
"""
from __future__ import annotations

from typing import Callable

from ..env import Env

DUMMY = "dummy_v0"


class Catalog:
    def __init__(self) -> None:
        self.factories: dict[str, Callable[[], Env]] = {}
        self.wire_ids: set[str] = set()  # organizer pack mappings in this catalog
        self.defaults: list[str] = []  # served when --tasks is not given (dummy is opt-in)

    def add(self, task_id: str, factory: Callable[[], Env], *, default: bool = True, wire_id: str = "") -> None:
        if task_id in self.factories:
            raise ValueError(f"duplicate task_id {task_id!r}")
        if wire_id:
            if wire_id in self.wire_ids:
                raise ValueError(f"duplicate tabletop wire_id {wire_id!r}")
            self.wire_ids.add(wire_id)
        self.factories[task_id] = factory
        if default:
            self.defaults.append(task_id)

    def make(self, task_id: str) -> Env:
        return self.factories[task_id]()


def builtin_catalog() -> Catalog:
    from .bench import TASKS as BENCH, BenchEnv
    from .cups import PUBLIC, CupsEnv
    from .dummy import DummyEnv
    from .placeholder import PUBLIC as PLACEHOLDERS, PlaceholderEnv
    from .tabletop import PUBLIC_IDS as TABLETOP, TabletopEnv

    catalog = Catalog()
    for task_id in TABLETOP:
        catalog.add(task_id, lambda i=task_id: TabletopEnv(i))
    for spec in PLACEHOLDERS:
        catalog.add(spec.task_id, lambda s=spec: PlaceholderEnv(s), default=False)
    # Bench lifts are the previous contest. They stay constructable and are not served.
    for task_id, obj in BENCH.items():
        catalog.add(task_id, lambda t=task_id, o=obj: BenchEnv(t, o), default=False)
    # Cup family stays constructable for packs and the scripted demo, and is not served.
    for task_id, params in PUBLIC.items():
        catalog.add(task_id, lambda t=task_id, p=params: CupsEnv(t, p), default=False)
    catalog.add(DUMMY, DummyEnv, default=False)
    return catalog


def make_env(task_id: str) -> Env:
    return builtin_catalog().make(task_id)
