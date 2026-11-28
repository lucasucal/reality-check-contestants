"""Standalone practice harness: drive an already-running policy server against an
already-running engine, with the SAME episode loop and aggregation as the real
runner (hidden-task aliasing off). This is what participants run locally.

    python -m rc_runner.practice --engine http://127.0.0.1:8090 --policy http://127.0.0.1:8080 \\
        [--tasks a,b] [--seeds 0-3] [--journal-dir DIR]

Waits for the policy's /readyz, sends the untimed warmup /act calls, then plays the episodes.
Prints per-episode lines and an /act timing summary to stderr and one JSON object on stdout:
{"scorer": "engine-v2", "total": ..., "breakdown": {task: mean episode points},
 "episodes": [one row per episode], "startup": {...}, "act_budget_s": 90.0} (see `report`).
Exit codes: 0 scored (a crash or timeout is an episode score, not a failed run),
2 setup/engine error, a failed warmup, or a task with no scored episodes.

This drives a policy and engine that are already running, wherever they run. Only
`python -m practice run` (contestants/PRACTICE.md) starts the policy in the published image
with the judged mounts and checks, and measures load time and memory.
"""
from __future__ import annotations

import argparse
import json
import math
import secrets
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO

from .engine_client import EngineClient
from .episode import ACT_BUDGET_EXCEEDED, PROTECTIVE_STOP, STEP_LIMIT, EpisodeResult, EpisodeSettings, build_envelope, run_episode
from . import launch
from .errors import EngineInfraError
from .journal import JournalWriter, episode_filename
from .motion import HOLD_REASONS, MotionViolation, clamp_action, clamp_lines
from .policy_client import ActResult, PolicyClient, PolicyFailure
from .scoring import (_percentile, aggregate, hidden_labels_for, points_text, require_engine_scorer, safety_stop_lines,
                      task_labels)


JUDGED_ACT_BUDGET_SECONDS = 90.0  # per-repetition /act budget of a judged run (RULES "Limits")
PRACTICE_TOOL_NOTE = ("note: this harness drives an already-running policy. Only `python -m practice run` "
                      "(contestants/PRACTICE.md) runs it in the published image with the judged mounts and checks.")


def act_timing(act_ms: list[float]) -> dict[str, Any]:
    """/act time of one repetition: delivered calls, their summed seconds (what the judged 90 s budget counts) and
    the nearest-rank p50/p95 in milliseconds (None without calls)."""
    ms = [float(x) for x in act_ms if isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0]
    if not ms:
        return {"calls": 0, "total_s": 0.0, "p50_ms": None, "p95_ms": None}
    return {"calls": len(ms), "total_s": round(sum(ms) / 1000.0, 3),
            "p50_ms": round(_percentile(ms, 50), 1), "p95_ms": round(_percentile(ms, 95), 1)}


def episode_rows(results: list[EpisodeResult]) -> list[dict[str, Any]]:
    """One machine-readable row per episode: what the stderr lines say, plus /act timing."""
    return [{"task": r.task_id, "seed": r.seed, "success": r.success, "score": r.score, "steps": r.steps,
             "quality": r.quality, "quality_limit": r.quality_limit,
             "end_reason": r.end_reason, "failure_code": r.failure_code,
             "clamps": {limit: dict(joints) for limit, joints in sorted(r.clamps.items())},
             "held": dict(sorted(r.held.items())), "act": act_timing(r.act_ms),
             "safety_peaks": dict(r.safety_peaks)}
            for r in results]


def report(score: dict[str, Any], results: list[EpisodeResult], startup: dict[str, Any] | None = None) -> dict[str, Any]:
    """The stdout JSON of both practice CLIs: the score keys unchanged (`scorer`, `total`, `breakdown`), plus
    `episodes` (episode_rows), `startup` ({"readyz_s", "warmup_s", "warmup_acts"} or None) and `act_budget_s`."""
    return {**score, "episodes": episode_rows(results), "startup": startup, "act_budget_s": JUDGED_ACT_BUDGET_SECONDS}


def act_summary_lines(results: list[EpisodeResult]) -> list[str]:
    """/act latency over every episode and /act seconds per repetition against the judged budget."""
    ms = [float(x) for r in results for x in r.act_ms]
    if not ms:
        return ["/act timing:", f"  {'/act calls':<13} none during repetitions"]
    totals = sorted(sum(r.act_ms) / 1000.0 for r in results)
    lines = ["/act timing:",
             f"  {'/act calls':<13} {len(ms)} calls, p50 {_percentile(ms, 50):.1f} ms, p95 {_percentile(ms, 95):.1f} ms, "
             f"max {max(ms):.1f} ms",
             f"  {'/act per rep':<13} median {statistics.median(totals):.1f} s, max {totals[-1]:.1f} s of the "
             f"{JUDGED_ACT_BUDGET_SECONDS:g} s judged budget ({len(totals)} repetitions)"]
    return lines


# Per-repetition table (stderr). Columns: task, seed, result, steps, points, notes; detail lines under a row.
RESULT_WIDTH = 16
POINTS_WIDTH = 7
DETAIL_INDENT = "      "
ENGINE_NEXT_STEP = ("start the public engine (docker run -d --rm --name rc-engine -p 127.0.0.1:8090:8090 rc-engine:dev, "
                    "see contestants/PRACTICE.md) and check that curl {url}/readyz answers, or pass --engine URL")


def repetition_header(width: int) -> str:
    return (f"  {'task':<{width}}  {'seed':>4}  {'result':<{RESULT_WIDTH}}  {'steps':>5}  {'points':>{POINTS_WIDTH}}"
            "  notes")


def repetition_line(result: EpisodeResult, *, label: str, width: int, act_budget: float) -> str:
    """One aligned row: "  cup_place_v0     0  success            412    83.79  precision 82% (limited by ...)"."""
    notes: list[str] = []
    if result.success:
        status = "success"
    elif result.failure_code:
        status = "FAILED"
        notes.append(f"{result.failure_code}: {result.message}" if result.message else result.failure_code)
    elif result.end_reason == ACT_BUDGET_EXCEEDED:
        status = "out of /act time"
        notes.append(f"the {act_budget:g} s /act budget ran out")
    elif result.end_reason == STEP_LIMIT:
        status = "step limit"
    elif result.end_reason == PROTECTIVE_STOP:
        status = "protective stop"
        notes.append("the engine enforced a UR12e safety limit (RC_SAFETY_STOP=enforce)")
    else:
        status = "no success"
    scored = result.score is not None and math.isfinite(result.score)
    points = points_text(result.score) if scored else "-"
    quality = result.quality if result.quality is not None and math.isfinite(result.quality) else None
    if result.success and scored and quality is not None:
        precision = f"precision {round(100 * min(max(quality, 0.0), 1.0)):d}%"
        if result.quality_limit and quality < 1:
            precision += f" (limited by {result.quality_limit})"
        notes.insert(0, precision)
    row = (f"  {label:<{width}}  {result.seed:>4}  {status:<{RESULT_WIDTH}}  {result.steps:>5}  "
           f"{points:>{POINTS_WIDTH}}")
    return (row + "  " + "; ".join(notes)) if notes else row


def repetition_details(result: EpisodeResult) -> list[str]:
    """Held steps, clamps, safety notices and ignored fields of one repetition (the judged page's wording)."""
    lines = []
    if result.held:
        counts = ", ".join(f"{HOLD_REASONS[k]}: {n}" for k, n in sorted(result.held.items()))
        lines.append(f"held position on {sum(result.held.values())} step(s) (malformed action; nothing from it was "
                     f"applied): {counts}")
    if result.clamps:  # same breakdown as the judged notice (CS-AC-10)
        lines.append(f"actions clamped: {'; '.join(clamp_lines(result.clamps, result.steps))}")
    stops = safety_stop_lines(result.safety_peaks)
    if stops:
        head = ("protective stop (enforced) ended it" if result.end_reason == PROTECTIVE_STOP
                else "would trigger a safety stop on the real arm")
        lines.append(f"{head}: {'; '.join(stops)}")
    if result.ignored_fields:
        lines.append(f"ignored fields: {', '.join(result.ignored_fields)}")
    return lines


def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def verdict_line(score: dict[str, Any] | None, results: list[EpisodeResult], failure: str | None = None) -> str:
    """The last stderr line of a practice run, e.g.
    "VERDICT: public mean 83.50 pts · 8 repetitions · 2 with clamps · 0 safety notices"."""
    total = (score or {}).get("total")
    if failure or not isinstance(total, (int, float)) or not math.isfinite(total):
        head = "VERDICT: no contest score" + (f" ({failure})" if failure else "")
    else:
        head = f"VERDICT: public mean {points_text(total)} pts"
    parts = [head, _count(len(results), "repetition"),
             f"{sum(1 for r in results if r.clamps)} with clamps",
             _count(sum(1 for r in results if safety_stop_lines(r.safety_peaks)), "safety notice")]
    held = sum(sum(r.held.values()) for r in results)
    if held:
        parts.append(_count(held, "held step"))
    failed = sum(1 for r in results if r.failure_code)
    if failed:
        parts.append(f"{failed} failed")
    return " · ".join(parts)


def parse_seeds(spec: str) -> list[int]:
    """'0-4' | '0,3,7' | '0-2,10' -> sorted unique ints."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"bad seed range {part!r}")
            seeds.update(range(lo_i, hi_i + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError("no seeds given")
    return sorted(seeds)


def _wait(label: str, probe: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe():
            return
        time.sleep(0.3)
    raise SystemExit(f"{label} not ready after {timeout:g}s")


def wait_for_engine(engine: EngineClient, timeout: float, url: str) -> None:
    """Wait for the engine's /readyz; on timeout exit with what was wrong and the one next step."""
    last: dict[str, Any] = {}

    def probe() -> bool:
        last.clear()
        last.update(engine.readyz())
        return bool(last.get("ready"))
    try:
        _wait("engine", probe, timeout)
    except SystemExit:
        if last.get("_unreachable"):
            why = f"nothing answers at {url}"
        elif last.get("error"):
            why = f"the engine at {url} reports an error: {last['error']}"
        else:
            why = f"the engine at {url} is up but not ready"
        raise SystemExit(f"engine not ready after {timeout:g}s: {why}. Next step: "
                         + ENGINE_NEXT_STEP.format(url=url.rstrip("/"))) from None


def practice(engine: EngineClient, policy: PolicyClient, *, tasks_filter: list[str] | None, seeds: list[int],
             journal_dir: Path | None, settings: EpisodeSettings, out: TextIO | None = None,
             episode_options: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[EpisodeResult], str | None]:
    """Run every (task, seed); returns (score dict, results, failure message or None).
    Same aggregation as the real runner, minus hidden aliasing. `episode_options`: the engine /reset `options`
    of every episode (practice: {"room": ...})."""
    out = out or sys.stderr
    require_engine_scorer(engine.readyz())
    schema_version, engine_tasks = engine.tasks()
    by_id = {t["task_id"]: t for t in engine_tasks}
    if tasks_filter:
        missing = [t for t in tasks_filter if t not in by_id]
        if missing:
            raise SystemExit(f"engine does not serve: {', '.join(missing)} (has: {', '.join(sorted(by_id))})")
        tasks = [by_id[t] for t in tasks_filter]
    else:
        tasks = [by_id[k] for k in sorted(by_id)]
    labels = task_labels(tasks, alias_hidden=False)
    hidden = hidden_labels_for(tasks, labels)
    results: list[EpisodeResult] = []
    ordinal = 0
    width = max([len("task"), *(len(labels[t["task_id"]]) for t in tasks)])
    header_printed = False
    for task in tasks:
        label = labels[task["task_id"]]
        for seed in seeds:
            ordinal += 1
            filename = episode_filename(ordinal)
            journal = JournalWriter(journal_dir / filename) if journal_dir else None
            try:
                result = run_episode(engine=engine, policy=policy, task=task, label=label, seed=seed, ordinal=ordinal,
                                     settings=settings, journal=journal, journal_file=filename if journal else None,
                                     schema_version=schema_version, episode_options=episode_options)
            finally:
                if journal:
                    journal.close()
            results.append(result)
            if not header_printed:
                print(repetition_header(width), file=out, flush=True)
                header_printed = True
            print(repetition_line(result, label=label, width=width,
                                  act_budget=min(settings.episode_timeout, JUDGED_ACT_BUDGET_SECONDS)),
                  file=out, flush=True)
            for detail in repetition_details(result):
                print(f"{DETAIL_INDENT}{detail}", file=out, flush=True)
    agg = aggregate(results, [labels[t["task_id"]] for t in tasks], hidden_labels=hidden)
    if agg.breakdown:
        print("Public task means:", file=out)
    for label, value in agg.breakdown.items():
        print(f"  {label:<{width}}  {points_text(value):>{POINTS_WIDTH}} pts", file=out)
    return agg.score(), results, (agg.failure_message if agg.failure_code else None)


def _policy_ready(policy: PolicyClient) -> Callable[[], bool]:
    def probe() -> bool:
        try:
            if policy.get("/healthz").status_code != 200:
                return False
            resp = policy.get("/readyz")
            payload = resp.json()
        except Exception:  # noqa: BLE001 - not up yet
            return False
        state = launch.readyz_state(resp.status_code, payload)
        if state == "load_error":
            raise SystemExit(f"policy /readyz reported a load error: {payload['error']}")
        if state == "contract_mismatch":
            raise SystemExit(
                f"policy /readyz contract/schema mismatch: expected contract='v0' schema_version=3, "
                f"got contract={payload.get('contract')!r} schema_version={payload.get('schema_version')!r}"
            )
        return state == "ready"
    return probe


class StartupFailure(Exception):
    """Startup/warmup failed with the runner's failure code (container_unhealthy, weights_load_failed,
    warmup_timeout, motion_limit_violation, ...)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def startup_and_warmup(engine: EngineClient, policy: PolicyClient, *, task_id: str, seed: int = 0, acts: int = 3,
                       ready_timeout: float = 1800.0, out: TextIO | None = None,
                       alive: Callable[[], bool] | None = None, started: float | None = None) -> dict[str, Any]:
    """The worker's startup sequence for practice: /healthz -> /readyz (a load error or a contract
    mismatch is fatal, as in judging) -> `acts` untimed synthetic /act calls on an engine /reset
    observation, each passed through the runner-side motion clamp. Mirrors worker._warmup's
    checks and failure codes; hardware excepted, passing this means judged startup passes.

    Returns {"readyz_s": seconds from `started` (time.monotonic(); default: this call) until /readyz was ready,
    "warmup_s": seconds the warmup /act calls took, "warmup_acts": acts}."""
    out = out or sys.stderr
    began = time.monotonic()
    started = began if started is None else started
    deadline = began + ready_timeout
    ready = _policy_ready(policy)
    print("waiting for policy /healthz and /readyz ...", file=out, flush=True)
    while True:
        if alive is not None and not alive():
            raise StartupFailure("container_unhealthy", "the policy container exited before it became ready")
        try:
            if ready():
                break
        except SystemExit as exc:
            raise StartupFailure("weights_load_failed", str(exc.code)) from exc
        if time.monotonic() >= deadline:
            raise StartupFailure("container_unhealthy", f"/healthz + /readyz did not pass within {ready_timeout:g}s")
        time.sleep(0.5)
    ready_at = time.monotonic()
    reset = engine.reset(task_id, seed)
    obs = reset["observation"]
    target = reset.get("simulator_state", {}).get("target", (obs.get("proprio") or {}).get("joint_pos", []))
    episode_id = f"warmup-{secrets.token_hex(6)}"
    step = int(obs.get("step", 0))
    body = build_envelope(obs, episode_id, step)
    for i in range(max(acts, 0)):
        print(f"warmup act {i + 1}/{acts}", file=out, flush=True)
        try:
            res = policy.act(body, episode_id=episode_id, step=step, token=f"warmup-{secrets.token_hex(8)}")
            if isinstance(target, list) and len(target) == 6:
                clamp_action(res.action, [float(x) for x in target])
        except MotionViolation as exc:  # judged runs hold position instead of failing
            print(f"warmup act {i + 1}: held position ({HOLD_REASONS[exc.reason]}); the action was not applied", file=out, flush=True)
        except PolicyFailure as exc:
            if exc.held_reason:
                print(f"warmup act {i + 1}: held position ({HOLD_REASONS[exc.held_reason]}); the action was not applied",
                      file=out, flush=True)
                continue
            code = "warmup_timeout" if exc.code in ("policy_timeout", "policy_crashed") else exc.code
            raise StartupFailure(code, f"warmup /act failed: {exc.message}") from exc
    return {"readyz_s": round(ready_at - started, 2), "warmup_s": round(time.monotonic() - ready_at, 2),
            "warmup_acts": max(acts, 0)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rc_runner.practice", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--engine", default="http://127.0.0.1:8090")
    parser.add_argument("--policy", default="http://127.0.0.1:8080")
    parser.add_argument("--tasks", default="", help="comma-separated task ids (default: every task the engine serves)")
    parser.add_argument("--seeds", default="0-3", help="e.g. 0-3 or 0,3,7-9 (default 0-3)")
    parser.add_argument("--journal-dir", default="", help="write episode-N.jsonl journals here")
    parser.add_argument("--act-timeout", type=float, default=10.0)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--warmup-acts", type=int, default=3,
                        help="untimed warmup /act calls after /readyz, as judging sends (default 3; 0 skips warmup)")
    parser.add_argument("--judged", action="store_true",
                        help="emulate judged evaluation settings (no legacy step cutoff, 90s /act budget unless overridden)")
    parser.add_argument("--scenario-seconds", type=float, default=90.0,
                        help="per-repetition cumulative /act budget in --judged mode (default 90s; not the 60 s simulated length)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="optional step cap in --judged mode (default: None, matching judged evaluation)")
    parser.add_argument("--camera-configuration", choices=("three_view",), default="three_view",
                        help="camera configuration passed to /reset in --judged mode")
    parser.add_argument("--room", default=None,
                        help="play every repetition in this room (e.g. real_cell, the venue); default: the seed's")
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    engine = EngineClient(args.engine)
    policy = PolicyClient(args.policy, act_timeout=args.act_timeout, connect_timeout=5.0, retries=0 if args.judged else 1)
    settings = (
        EpisodeSettings(
            episode_timeout=args.scenario_seconds,
            judged=True,
            max_steps=args.max_steps,
            camera_configuration=args.camera_configuration,
        )
        if args.judged
        else EpisodeSettings(episode_timeout=args.episode_timeout)
    )
    print(PRACTICE_TOOL_NOTE, file=sys.stderr, flush=True)
    tasks_filter = [t for t in args.tasks.split(",") if t] or None
    try:
        started = time.monotonic()
        wait_for_engine(engine, args.ready_timeout, args.engine)
        _wait("policy", _policy_ready(policy), args.ready_timeout)
        startup: dict[str, Any] = {"readyz_s": round(time.monotonic() - started, 2), "warmup_s": None, "warmup_acts": 0}
        if args.warmup_acts > 0:
            warm_task = tasks_filter[0] if tasks_filter else min(t["task_id"] for t in engine.tasks()[1])
            try:
                warm = startup_and_warmup(engine, policy, task_id=warm_task, acts=args.warmup_acts,
                                          ready_timeout=args.ready_timeout)
            except StartupFailure as exc:
                print(f"startup failed ({exc.code}): {exc.message}", file=sys.stderr)
                print(f"VERDICT: startup failed ({exc.code}) · no score", file=sys.stderr)
                print(json.dumps({"scorer": None, "total": None, "error": str(exc), "breakdown": {},
                                  "episodes": [], "startup": startup, "act_budget_s": JUDGED_ACT_BUDGET_SECONDS}))
                return 2
            startup.update(warmup_s=warm["warmup_s"], warmup_acts=warm["warmup_acts"])
        score, results, failure = practice(
            engine, policy, tasks_filter=tasks_filter, seeds=seeds,
            journal_dir=Path(args.journal_dir) if args.journal_dir else None,
            settings=settings, episode_options={"room": args.room} if args.room else None)
    except EngineInfraError as exc:
        print(f"engine error (not your policy's fault): {exc}", file=sys.stderr)
        print("Next step: " + ENGINE_NEXT_STEP.format(url=args.engine.rstrip("/")), file=sys.stderr)
        print("VERDICT: engine error · no score", file=sys.stderr)
        return 2
    finally:
        engine.close()
        policy.close()
    for line in act_summary_lines(results):
        print(line, file=sys.stderr)
    if failure:
        print(f"no contest score: {failure}", file=sys.stderr)
        print(verdict_line(score, results, failure), file=sys.stderr)
        print(json.dumps(report({"scorer": score["scorer"], "total": None, "error": failure,
                                 "breakdown": score["breakdown"]}, results, startup)))
        return 2
    print(verdict_line(score, results), file=sys.stderr)
    print(json.dumps(report(score, results, startup)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
