"""Contest aggregation and hidden-task aliasing.

Episode scores come from the engine (`engine/rc_engine/score.py`, scorer engine-v2) in points: completeness,
finish precision and time, no energy. A completed repetition scores 70 + Q * (5 + B), where Q is the finish
quality (0..1: full credit inside an ideal band around the seated pose, falling linearly to 0 at the pass
tolerance, the minimum over measurements, parts and hold samples) and B = 10 * ln(T / t) / ln(2) is the speed
bonus (T = step budget, t = steps used). There is no cap: a fast, precise run can exceed 100. A repetition that
did not complete scores 15/70 of the engine-v1 partial-progress points, always below 15. A task score is the mean
of its episode scores. An infra attempt needs a replacement score for the same seed. A policy crash or timeout
stays in the mean: the engine's failure score when the episode carries one, otherwise 0. The contest score is
the weighted mean of the task scores over the tasks the run was configured with: weight 1 per public task, 1.5
per hidden task, so the denominator is n_public + 1.5 * n_hidden (`contest_weight_sum`). The default two public
and four hidden tasks give 8:

    (p1 + p2 + 1.5 * (h1 + h2 + h3 + h4)) / 8

The number is in points (>= 0, no upper bound), and one public task's contest number is that task's mean.
There is no N-of-M gate and no tie-break.

The published score keeps public task names and each hidden task's score under its anonymous alias
(hidden_1..hidden_4, in hidden task_id order, so hidden_N is wire id N); raw wire ids are dropped. It does
not carry doneness, speed, energy (always 0.0 on the engine wire), centimeters, or seconds. Public evaluation
summary rows carry each public repetition's points, finish quality and limiting measurement; hidden repetitions
never do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import re
from decimal import ROUND_HALF_UP, Decimal

from .episode import EpisodeResult
from .errors import EngineInfraError
from .motion import CLAMP_JOINTS, CLAMP_LIMITS
from .policy_client import DEFAULT_MAX_RESPONSE_BYTES

SCORER = "engine-v2"
WIRE_IDS = {"1", "2", "3", "4"}  # hidden tasks on the engine wire (engine/rc_engine/packs.py)
PUBLIC_WEIGHT = 1.0
HIDDEN_WEIGHT = 1.5
_HIDDEN_ALIAS = re.compile(r"hidden_[1-4]")


def require_engine_scorer(payload: dict[str, Any]) -> None:
    """Require the engine to advertise the formula this runner publishes."""
    if payload.get("scorer") != SCORER:
        raise EngineInfraError(f"engine must advertise {SCORER} in /readyz")


def hidden_report_label(label: str) -> bool:
    """True when publishing this label would say which hidden task it was."""
    return label in WIRE_IDS or label.startswith("hidden_")


def task_labels(tasks: list[dict[str, Any]], *, alias_hidden: bool = True) -> dict[str, str]:
    """task_id -> reporting label. Public (or visibility-less) tasks keep their id; hidden
    ones become hidden_1.. in task_id order when `alias_hidden` (off for practice)."""
    labels: dict[str, str] = {}
    hidden = sorted(t["task_id"] for t in tasks if alias_hidden and t.get("visibility") == "hidden")
    aliases = {task_id: f"hidden_{i}" for i, task_id in enumerate(hidden, start=1)}
    for task in tasks:
        labels[task["task_id"]] = aliases.get(task["task_id"], task["task_id"])
    return labels


def hidden_labels_for(tasks: list[dict[str, Any]], labels: dict[str, str]) -> set[str]:
    """Reporting labels the pack marks hidden, plus the hidden wire ids (WIRE_IDS)."""
    hidden: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        label = labels.get(task_id, task_id)
        if task.get("visibility") == "hidden" or hidden_report_label(label) or hidden_report_label(task_id):
            hidden.add(label)
    return hidden


def _is_hidden(label: str, hidden_labels: set[str]) -> bool:
    return label in hidden_labels or hidden_report_label(label)


def _weight(label: str, hidden_labels: set[str]) -> float:
    return HIDDEN_WEIGHT if _is_hidden(label, hidden_labels) else PUBLIC_WEIGHT


def contest_weight_sum(labels: list[str], hidden_labels: set[str] | None = None) -> float:
    """The contest denominator for these task labels: n_public * PUBLIC_WEIGHT + n_hidden * HIDDEN_WEIGHT."""
    hidden = set(hidden_labels or ())
    return sum(_weight(label, hidden) for label in dict.fromkeys(labels))


def _subject(label: str, hidden_labels: set[str]) -> str:
    if _is_hidden(label, hidden_labels):
        return "a task"
    return f"task {label}"


@dataclass
class Aggregate:
    total: float | None
    breakdown: dict[str, float]
    failure_code: str | None = None
    failure_message: str = ""

    def score(self) -> dict[str, Any]:
        """Team-facing score: public task scores and hidden task scores under their hidden_N aliases. Hidden wire
        ids are omitted. Metrics are not included."""
        shown = {label: value for label, value in self.breakdown.items()
                 if not hidden_report_label(label) or _HIDDEN_ALIAS.fullmatch(label)}
        return {"scorer": SCORER, "total": self.total, "breakdown": shown}


def _episode_value(result: EpisodeResult) -> tuple[float | None, str]:
    """(number, kind). Infra is excluded. A team failure with no engine score is 0.
    A finished episode with no engine score is a gap: the boolean success bit is not a score."""
    if result.failure_code == "infra_error":
        return None, "infra"
    if result.score is not None:
        return float(result.score), "score"
    if result.failure_code or not result.completed:
        return 0.0, "score"
    return None, "gap"


def _task_mean(task_results: list[EpisodeResult]) -> tuple[float | None, str | None]:
    values: dict[tuple[int, int | None], float] = {}
    interrupted: set[tuple[int, int | None]] = set()
    gap = False
    for result in task_results:
        identity = (result.seed, result.repetition)
        value, kind = _episode_value(result)
        if kind == "infra":
            interrupted.add(identity)
            continue
        if kind == "gap":
            gap = True
            continue
        if value is not None:
            if identity in values:
                return None, "duplicate_score"
            values[identity] = value
    if gap:
        return None, "missing_score"
    if interrupted.difference(values) or not values:
        return None, "unscored"
    return round(sum(values.values()) / len(values), 6), None


def aggregate(
    results: list[EpisodeResult],
    labels_in_order: list[str],
    *,
    hidden_labels: set[str] | None = None,
) -> Aggregate:
    """Task score = mean of scored episodes. Contest score = weighted mean of task scores.

    `hidden_labels` are reporting labels (practice keeps real ids; the official runner
    passes the hidden_* aliases). Wire ids and hidden_* labels are hidden either way.
    """
    hidden = set(hidden_labels or ())
    ordered: list[str] = []
    for label in labels_in_order:
        if label not in ordered:
            ordered.append(label)
    breakdown: dict[str, float] = {}
    problems: list[tuple[str, str]] = []
    for label in ordered:
        task_results = [result for result in results if result.task_id == label]
        mean, problem = _task_mean(task_results)
        if problem is not None or mean is None:
            problems.append((label, problem or "unscored"))
            continue
        breakdown[label] = mean
    if problems:
        parts: list[str] = []
        for label, problem in problems:
            subject = _subject(label, hidden)
            if problem == "missing_score":
                parts.append(f"{subject}: engine episode result has no score; the runner does not invent one")
            elif problem == "duplicate_score":
                parts.append(f"{subject}: duplicate episode score for a seed (not your fault; resubmit)")
            else:
                parts.append(
                    f"{subject}: missing episode scores after infra retries were exhausted; "
                    "the spec does not define a number (not your fault; resubmit)")
        return Aggregate(None, breakdown, "infra_error", "; ".join(parts))
    if not breakdown:
        return Aggregate(None, {}, "infra_error", "the spec does not define a number (not your fault; resubmit)")
    weight_sum = contest_weight_sum(ordered, hidden)
    weighted = sum(breakdown[label] * _weight(label, hidden) for label in ordered)
    total = round(weighted / weight_sum, 6)
    return Aggregate(total, breakdown)


def min_completed(episodes_in_task: int) -> int:
    """Unused. The runner no longer applies an N-of-M completion gate."""
    return (episodes_in_task + 1) // 2


def task_violates_n_of_m(results: list[EpisodeResult], total_episodes: int) -> bool:
    """Unused. The worker does not fail fast, and aggregate does not reject a submission."""
    completed = sum(1 for result in results if result.completed)
    remaining = total_episodes - len(results)
    return completed + remaining < min_completed(total_episodes)


def task_failure(label: str, task_results: list[EpisodeResult], breakdown: dict[str, float] | None = None) -> Aggregate:
    """Unused. A crash or timeout is an episode score, not a submission failure."""
    completed = sum(1 for result in task_results if result.completed)
    first = next((result for result in task_results if result.failure_code), task_results[0])
    shown = {key: value for key, value in (breakdown or {}).items() if not hidden_report_label(key)}
    subject = "a task" if hidden_report_label(label) else f"task {label}"
    return Aggregate(
        None, shown, first.failure_code or "policy_timeout",
        f"{subject}: only {completed}/{len(task_results)} episodes completed; "
        f"first failure: {first.failure_code} ({first.message})".strip())


# ------------------------------------------------------------------ evaluation_summary (team-facing)
# docs/verification/error-surfacing-contract.md: a whitelist projector over the episode results. Public rows carry
# exactly these keys (never `message`, digests, journals or seeds); hidden episodes collapse to one boolean and
# aggregate category counts (DEC-6);
# the notes are computed from public episodes only.
SUMMARY_ROW_KEYS = ("task", "repetition", "success", "steps", "end_reason", "failure_code", "actions_clamped",
                    "score", "quality", "quality_limit")
QUALITY_LIMIT_RE = re.compile(r"^[A-Za-z0-9_ .,:/()+-]{1,80}$")
# Optional public row key: a closed, static reason code for an episode the runner ended (BAD-06). Present only when
# set. Values are fixed words, never policy text. Hidden episodes never carry it (they only add to `ended_early`).
ROW_REASONS = {"too_large": "response_too_large"}  # EpisodeResult.failure_reason -> row `reason`
# Episode failure codes that are not the policy's doing; anything else is an abnormal (policy-side) ending.
_NOT_ABNORMAL = frozenset({None, "", "infra_error"})
# DEC-6: hidden episodes that ended early, as counts per category, aggregated over ALL hidden tasks and repetitions
# (no task, repetition, step, message or timing). A closed key set, always all present.
HIDDEN_END_CATEGORIES = ("policy_crashed", "policy_timeout", "policy_error", "act_budget_exceeded")
_CRASH_CODES = frozenset({"policy_crashed", "out_of_memory", "container_unhealthy"})


def _hidden_end_category(result: EpisodeResult) -> str | None:
    if result.failure_code in _CRASH_CODES:
        return "policy_crashed"
    if result.failure_code == "policy_timeout":
        return "policy_timeout"
    if result.failure_code not in _NOT_ABNORMAL:
        return "policy_error"
    return "act_budget_exceeded" if result.end_reason == "act_budget_exceeded" else None


def hidden_summary(hidden_results: list[EpisodeResult]) -> dict[str, Any]:
    """The team-facing hidden block: the abnormal-ending flag and the DEC-6 category counts."""
    counts = dict.fromkeys(HIDDEN_END_CATEGORIES, 0)
    for r in hidden_results:
        category = _hidden_end_category(r)
        if category:
            counts[category] += 1
    return {"abnormal_endings": any(r.failure_code not in _NOT_ABNORMAL for r in hidden_results), "ended_early": counts}


def _end_reason(result: EpisodeResult) -> str:
    """Closed set: act_budget_exceeded, step_limit, done, policy_error, infra_error, incomplete."""
    if result.end_reason in ("act_budget_exceeded", "step_limit"):
        return result.end_reason
    if result.failure_code == "infra_error":
        return "infra_error"
    if result.failure_code:
        return "policy_error"
    return "done" if result.completed else "incomplete"


def _public_results(results: list[EpisodeResult], hidden: set[str]) -> list[EpisodeResult]:
    return [r for r in results if not hidden_report_label(r.task_id) and r.task_id not in hidden]


POINTS_DECIMALS = 2  # equals backend routes.SCORE_DECIMALS: points are shown at the leaderboard precision


def round_points(value: float) -> float:
    """Points rounded half up to POINTS_DECIMALS (Decimal on the shortest repr, as the backend leaderboard does)."""
    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-POINTS_DECIMALS), rounding=ROUND_HALF_UP))


def points_text(value: float) -> str:
    """"83.79": points at the leaderboard precision, half up."""
    if not math.isfinite(value):
        return str(value)
    return f"{round_points(value):.{POINTS_DECIMALS}f}"


def _points(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return round_points(value)


def _quality(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(min(max(float(value), 0.0), 1.0), 4)


def _limit_text(value: str | None) -> str | None:
    return value if isinstance(value, str) and QUALITY_LIMIT_RE.match(value) else None


def public_rows(results: list[EpisodeResult], hidden_labels: set[str] | None = None) -> list[dict[str, Any]]:
    """The whitelisted team-facing rows (SUMMARY_ROW_KEYS) of the public episodes only.
    Shared by evaluation_summary and the live `progress.public.episodes`."""
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for r in _public_results(results, set(hidden_labels or ())):
        seen[r.task_id] = seen.get(r.task_id, 0) + 1
        repetition = r.repetition if isinstance(r.repetition, int) and not isinstance(r.repetition, bool) else seen[r.task_id]
        row = {"task": str(r.task_id), "repetition": repetition, "success": bool(r.success),
               "steps": int(r.steps), "end_reason": _end_reason(r),
               "failure_code": r.failure_code or None, "actions_clamped": bool(r.actions_clamped),
               "score": _points(r.score), "quality": _quality(r.quality), "quality_limit": _limit_text(r.quality_limit)}
        reason = ROW_REASONS.get(r.failure_reason or "") if r.failure_code else None
        if reason:
            row["reason"] = reason
        rows.append(row)
    return rows


GIB = 1024 ** 3
MAX_LOAD_STARTS = 16  # load_seconds lists at most this many public container starts (the first and the latest)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100) of a non-empty list (same rule as the practice resource summary)."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def public_clamps(public: list[EpisodeResult]) -> dict[str, dict[str, int]]:
    """{limit: {joint: steps}} summed over public episodes, closed key sets (motion.CLAMP_LIMITS / CLAMP_JOINTS)."""
    out: dict[str, dict[str, int]] = {}
    for limit in CLAMP_LIMITS:
        for joint in CLAMP_JOINTS[limit]:
            n = sum(int((r.clamps.get(limit) or {}).get(joint, 0)) for r in public)
            if n:
                out.setdefault(limit, {})[joint] = n
    return out


SAFETY_STOP_LIMITS = {"tcp_force": ("tcp_force_n", 150.0), "momentum": ("momentum_kg_m_s", 25.0),
                      "power": ("power_w", 300.0)}


def public_safety_stops(public: list[EpisodeResult]) -> dict[str, int]:
    """{"of", "tcp_force", "momentum", "power"}: public repetitions whose peak went over each UR12e safety limit
    (they would trigger a safety stop on the real arm). Empty when none did. The limits are the UR12e's default
    (Normal) safety-mode TCP force, momentum and power; the engine measures the peaks (arm_limits.py), nothing
    enforces them and they are not scored."""
    counts = {key: sum(1 for r in public if float(r.safety_peaks.get(peak, 0.0)) > limit)
              for key, (peak, limit) in SAFETY_STOP_LIMITS.items()}
    return {"of": len(public), **counts} if any(counts.values()) else {}


def safety_stop_lines(peaks: dict[str, float]) -> list[str]:
    """Team-facing lines for one repetition's peaks over a UR12e safety limit, e.g. "TCP force peak 182 N (limit 150 N)"."""
    names = {"tcp_force": ("TCP force", "N"), "momentum": ("momentum", "kg·m/s"), "power": ("power", "W")}
    lines = []
    for key, (peak, limit) in SAFETY_STOP_LIMITS.items():
        value = float(peaks.get(peak, 0.0))
        if value > limit:
            name, unit = names[key]
            lines.append(f"{name} peak {value:.0f} {unit} (limit {limit:.0f} {unit})")
    return lines


def public_resources(public: list[EpisodeResult], *, load_seconds: list[float] | None = None,
                     memory_peak_bytes: int | None = None) -> dict[str, Any]:
    """CS-AC-15: the policy's own resource use in the public phase only. load_seconds and memory_peak_bytes come
    from the worker, which stops recording them when the hidden phase begins; /act latency is computed here from
    public episodes. Rounded, bounded numbers; null when not measured."""
    loads = [round(min(max(float(x), 0.0), 86400.0), 1) for x in (load_seconds or [])]
    if len(loads) > MAX_LOAD_STARTS:
        loads = loads[:1] + loads[-(MAX_LOAD_STARTS - 1):]
    samples = [float(x) for r in public for x in r.act_ms if isinstance(x, (int, float)) and x >= 0]
    act = ({"calls": len(samples), "p50": round(min(_percentile(samples, 50), 600000.0), 1),
            "p95": round(min(_percentile(samples, 95), 600000.0), 1)} if samples else None)
    peak = (round(min(memory_peak_bytes / GIB, 4096.0), 1)
            if isinstance(memory_peak_bytes, int) and memory_peak_bytes > 0 else None)
    return {"load_seconds": loads, "memory_peak_gib": peak, "act_ms": act}


def evaluation_summary(results: list[EpisodeResult], hidden_labels: set[str] | None = None, *,
                       held_public: dict[str, int] | None = None, held_hidden: int | None = None,
                       ignored_fields: list[str] | None = None,
                       response_cap_bytes: int | None = None,
                       load_seconds: list[float] | None = None,
                       memory_peak_bytes: int | None = None) -> dict[str, Any]:
    hidden = set(hidden_labels or ())
    public = _public_results(results, hidden)
    hidden_results = [r for r in results if hidden_report_label(r.task_id) or r.task_id in hidden]
    rows = public_rows(results, hidden)
    of = len(public)
    over = [r for r in public if r.end_reason == "act_budget_exceeded"]
    over_steps = sorted(int(r.steps) for r in over if r.steps > 0)
    if held_public is None:  # derive from the episodes when the caller (e.g. a test) gives no worker totals
        held_public = {}
        for r in public:
            for reason, n in r.held.items():
                held_public[reason] = held_public.get(reason, 0) + n
    if held_hidden is None:
        held_hidden = sum(sum(r.held.values()) for r in hidden_results)
    extra: dict[str, Any] = {}
    if any(held_public.values()) or held_hidden:
        # Public counts by reason; hidden episodes only one aggregate count (no episode, task, step or time).
        extra["held_steps"] = {"episodes": sum(1 for r in public if r.held), "of": of,
                               **{k: int(v) for k, v in sorted(held_public.items()) if v}, "hidden": int(held_hidden)}
    # H1: only names from public episodes; the worker's list is already restricted to the pre-hidden phase.
    names = sorted(ignored_fields) if ignored_fields is not None else sorted({n for r in public for n in r.ignored_fields})
    if names:
        extra["ignored_fields"] = {"names": names}
    # BAD-06: public episodes the runner ended because one /act response was over the size cap (static cap value).
    too_large = sum(1 for row in rows if row.get("reason") == "response_too_large")
    if too_large:
        extra["response_too_large"] = {"count": too_large, "of": of,
                                       "limit_bytes": int(response_cap_bytes or DEFAULT_MAX_RESPONSE_BYTES)}
    stops = public_safety_stops(public)
    if stops:
        extra["safety_stop"] = stops
    return {
        "public": rows,
        "hidden": hidden_summary(hidden_results),
        "resources": public_resources(public, load_seconds=load_seconds, memory_peak_bytes=memory_peak_bytes),
        "notes": {**extra,
            "act_budget_exceeded": {"count": len(over), "of": of,
                                    "before_first_step": sum(1 for r in over if r.steps == 0),
                                    "median_steps": over_steps[len(over_steps) // 2] if over_steps else None},
            "step_limit": {"count": sum(1 for r in public if r.end_reason == "step_limit"), "of": of},
            # CS-AC-10: which limit clamped which joint, in applied steps of public repetitions only.
            "clamped": {"episodes": sum(1 for r in public if r.actions_clamped), "of": of,
                        "steps_of": sum(int(r.steps) for r in public), "limits": public_clamps(public)},
        },
    }
