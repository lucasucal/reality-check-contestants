"""The lockstep episode loop (plan §4) — ONE implementation shared by the real
worker and the practice CLI.

Per control step: build the /act envelope (the engine's observation object plus
`episode_id`, base64 images embedded verbatim — no decode/re-encode), POST it to
the policy with X-RC-Episode-Id/Step/Token headers (single byte-identical retry on
timeout/connect failure), verify the echo, forward `action` verbatim to the
engine's /step with `step` for retry-safety, journal one JSONL line.

Failure attribution: PolicyFailure / budget overrun end the episode as a TEAM
failure (returned in the result). An action the engine rejects (ActionRejected)
is replaced by one hold-position step (DEC-2); the engine rejecting a hold step
is an EngineInfraError. A team failure keeps the
engine score when the episode already has one; otherwise score is 0 and
doneness is 0. EngineInfraError is raised — the caller discards the episode,
restarts the engine and retries once. A second infra failure is not a score.
"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .engine_client import EngineClient
from .errors import ActionRejected, EngineInfraError
from .journal import (JOURNAL_FORMAT, compact_state, JournalWriter, evidence_response, image_digest, image_digests, now_iso,
                      state_chain_digest)
from .judged_plan import engine_options
from .policy_client import PolicyClient, PolicyFailure
from .motion import MotionViolation, add_clamps, add_speed_scaled, clamp_action, hold_action, ignored_top_level, note_ignored


ACT_BUDGET_EXCEEDED = "act_budget_exceeded"
STEP_LIMIT = "step_limit"
DEFAULT_STEP_LIMIT_SECONDS = 60.0
_OMIT_OVER = 4096  # strings longer than this (base64 frames) are journaled as hash + size on non-image steps


def step_limit_steps(seconds: Any, control_hz: Any) -> int | None:
    """Simulated-time cap of one judged repetition in control steps (seconds x the task's control rate)."""
    if seconds is None or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)) or control_hz <= 0:
        return None
    return max(1, math.ceil(seconds * control_hz - 1e-9))


def slim(value: Any) -> Any:
    """Copy of a journal payload with long strings (base64 images) replaced by a size and hash marker."""
    if isinstance(value, str):
        if len(value) > _OMIT_OVER:
            return f"<omitted {len(value)} chars sha256:{hashlib.sha256(value.encode()).hexdigest()[:16]}>"
        return value
    if isinstance(value, dict):
        return {k: slim(v) for k, v in value.items()}
    if isinstance(value, list):
        return [slim(v) for v in value]
    return value


@dataclass
class EpisodeResult:
    ordinal: int
    task_id: str  # the label used for reporting/journals (alias for hidden tasks)
    seed: int
    # Judged plan options sent to the engine /reset ({"profile", "nudge"}); None outside judged runs. Organizer-only.
    episode_options: dict[str, Any] | None = None
    episode_id: str | None = None
    completed: bool = False  # the episode ran to engine `done` (team failures are not completed)
    success: bool = False
    score: float | None = None  # engine episode score; not part of the published summary
    doneness: float | None = None
    speed: float | None = None
    energy: float | None = None
    quality: float | None = None
    quality_limit: str | None = None
    failure_code: str | None = None
    message: str = ""
    steps: int = 0
    digest: str = ""
    journal_file: str | None = None
    substep_hashes: list[str] = field(default_factory=list, repr=False)
    repetition: int | None = None
    policy_pending: bool = field(default=False, repr=False)
    actions_clamped: bool = False
    # Why a *completed* repetition ended early, for the team. Not a failure: it never changes scoring.
    # "act_budget_exceeded": the policy's /act time used up the per-repetition budget (scenario clock).
    # "step_limit": the repetition ran its full simulated-time cap (default 60 simulated seconds).
    end_reason: str | None = None
    team_detail: str | None = field(default=None, repr=False)  # team-safe one-liner from the PolicyFailure
    failure_reason: str | None = field(default=None, repr=False)  # error-surfacing reason (not in summary())
    # Hold-position steps by motion.HOLD_REASONS id (malformed action answered with zero delta + previous gripper).
    held: dict[str, int] = field(default_factory=dict, repr=False)
    ignored_fields: list[str] = field(default_factory=list, repr=False)  # unknown top-level response/action keys
    # Clamped applied steps by limit and joint: {motion.CLAMP_LIMITS id: {joint or "gripper": steps}} (CS-AC-10).
    clamps: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    # Wall milliseconds of each delivered /act response (the answering attempt), for the public resource numbers.
    act_ms: list[float] = field(default_factory=list, repr=False)
    # Judged: /act seconds counted against the repetition budget (run_timing), the failed call included.
    policy_seconds: float = field(default=0.0, repr=False)
    safety_peaks: dict[str, float] = field(default_factory=dict, repr=False)

    def summary(self) -> dict[str, Any]:
        """Entry for the score event's `episodes` list (labels only; no hidden names)."""
        out: dict[str, Any] = {
            "ordinal": self.ordinal, "task": self.task_id, "seed": self.seed,
            "success": self.success, "completed": self.completed, "steps": self.steps,
            "state_chain_digest": self.digest, "journal": self.journal_file,
            "actions_clamped": self.actions_clamped,
        }
        if self.failure_code:
            out["failure_code"] = self.failure_code
        if self.end_reason:
            out["end_reason"] = self.end_reason
        if self.repetition is not None:
            out['repetition'] = self.repetition
        if self.episode_options is not None:
            out['episode_options'] = dict(self.episode_options)
        if self.safety_peaks:
            out['safety_peaks'] = dict(self.safety_peaks)
        return out


SAFETY_PEAK_KEYS = ("tcp_force_n", "momentum_kg_m_s", "power_w")


def _note_safety(result: EpisodeResult, info: Any) -> None:
    """Keep the largest `info.safety_peaks` value per key (finite, non-negative numbers only)."""
    peaks = info.get("safety_peaks") if isinstance(info, dict) else None
    if not isinstance(peaks, dict):
        return
    for key in SAFETY_PEAK_KEYS:
        value = peaks.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            if value > result.safety_peaks.get(key, 0.0) or key not in result.safety_peaks:
                result.safety_peaks[key] = float(value)


def _engine_target(response: Any) -> list[float] | None:
    """The joint target the engine holds after a step (simulator_state.target), when it reports a valid one."""
    state = response.get("simulator_state") if isinstance(response, dict) else None
    target = state.get("target") if isinstance(state, dict) else None
    if (isinstance(target, list) and len(target) == 6
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in target)):
        return [float(x) for x in target]
    return None


_METRICS = ("score", "doneness", "speed", "energy", "quality")
QUALITY_LIMIT_MAX_CHARS = 80


def _metric(payload: dict[str, Any], key: str) -> float | None:
    """Read a numeric episode field if the engine sent one. Do not invent it."""
    sources: list[dict[str, Any]] = [payload]
    info = payload.get("info")
    if isinstance(info, dict):
        sources.append(info)
    for source in sources:
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def _capture_metrics(result: EpisodeResult, payload: dict[str, Any]) -> None:
    for key in _METRICS:
        value = _metric(payload, key)
        if value is not None:
            setattr(result, key, value)
    limit = _quality_limit(payload)
    if limit is not _ABSENT:
        result.quality_limit = limit


_ABSENT = object()


def _quality_limit(payload: dict[str, Any]) -> Any:
    sources: list[dict[str, Any]] = [payload]
    info = payload.get("info")
    if isinstance(info, dict):
        sources.append(info)
    for source in sources:
        if "quality_limit" not in source:
            continue
        value = source["quality_limit"]
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()[:QUALITY_LIMIT_MAX_CHARS]
    return _ABSENT


def _team_failure(result: EpisodeResult, code: str, message: str) -> None:
    """A crash, timeout, or other team failure. Keep an engine score; otherwise 0."""
    result.failure_code = code
    result.message = message
    result.success = False
    if result.score is None:
        result.score = 0.0
    if result.doneness is None:
        result.doneness = 0.0


@dataclass
class EpisodeSettings:
    episode_timeout: float = 900.0
    final_hold: float = 1.5  # seconds to hold the final frame when pace > 0 (demo viewing)
    judged: bool = False
    max_steps: int | None = None
    # Judged repetitions also end after this many simulated seconds (steps = seconds x control_hz); None disables.
    step_limit_seconds: float | None = DEFAULT_STEP_LIMIT_SECONDS
    # Full image bodies in the journal on step 0 and every Nth step (0 = never, the default; 1 = every step).
    # Whatever the value, every step keeps simulator state plus sha256/size of each image sent to the policy.
    journal_image_every: int = 0
    engine_info: dict[str, Any] | None = None  # engine image id + renderer, written to the journal header
    camera_configuration: str = 'three_view'
    camera_poses: dict[str, Any] | None = None


class FramePublisher(Protocol):
    def publish(self, frame: dict[str, Any], force: bool = False) -> None: ...


WAIT_SLICE = 0.05  # paced waits sleep in slices so check() (lease/stop) stays responsive


class _PolicyDeadline(Exception):
    pass


def _bounded_act(policy: PolicyClient, body: bytes, *, episode_id: str, step: int, token: str,
                 budget: float, check: Callable[[], None] | None,
                 record: Callable[[dict[str, Any]], None] | None) -> Any:
    """Wait at most budget even if the HTTP peer continuously trickles bytes.

    A timed-out request can still be running in its daemon thread; its container
    must be restarted before another episode uses that policy.
    """
    finished = threading.Event()
    request_finished = threading.Event()
    request_start: list[float] = []
    request_durations: list[float] = []
    def mark_request_finished(duration: float) -> None:
        request_durations.append(duration)
        request_finished.set()
    outcome: list[Any] = []
    records: list[dict[str, Any]] = []
    def invoke() -> None:
        try:
            outcome.append(policy.act(body, episode_id=episode_id, step=step, token=token,
                                      timeout=budget, retries=0, record=records.append,
                                      request_finished=mark_request_finished, request_started=request_start.append))
        except BaseException as exc:
            outcome.append(exc)
        finally:
            finished.set()
    threading.Thread(target=invoke, name='rc-policy-act', daemon=True).start()
    while True:
        if check:
            check()
        remaining = request_start[0] + budget - time.monotonic() if request_start else WAIT_SLICE
        if remaining <= 0 and not request_finished.is_set():
            if record:
                record({'error': 'absolute scenario deadline during policy request'})
            raise _PolicyDeadline()
        if finished.wait(WAIT_SLICE if request_finished.is_set() else max(0, min(remaining, WAIT_SLICE))):
            if record:
                for item in records:
                    record(item)
            value = outcome[0]
            if isinstance(value, BaseException):
                if isinstance(value, PolicyFailure) and request_durations:
                    value.request_seconds = request_durations[-1]
                raise value
            return value


def _wait(seconds: float, check: Callable[[], None] | None, clock: Callable[[], float],
          sleep: Callable[[float], None]) -> None:
    """Interruptible wall-clock wait: check() runs between slices and may raise."""
    end = clock() + seconds
    while True:
        if check:
            check()
        remaining = end - clock()
        if remaining <= 0:
            return
        sleep(min(WAIT_SLICE, remaining))


def build_envelope(observation: dict[str, Any], episode_id: str, step: int) -> bytes:
    """Serialize the /act body once; the same bytes are reused for the retry and hashed
    for the journal. Image strings are carried through untouched."""
    envelope = {**observation, "episode_id": episode_id}
    envelope.setdefault("step", step)
    return json.dumps(envelope, separators=(",", ":")).encode()


def packed(response: dict[str, Any]) -> dict[str, Any]:
    """A full-image response with its simulator_state packed (images stay)."""
    if isinstance(response.get("simulator_state"), dict):
        return {**response, "simulator_state": compact_state(response["simulator_state"])}
    return response


def run_episode(
    *,
    engine: EngineClient,
    policy: PolicyClient,
    task: dict[str, Any],
    label: str,
    seed: int,
    ordinal: int,
    settings: EpisodeSettings,
    journal: JournalWriter | None = None,
    journal_file: str | None = None,
    submission_id: int | None = None,
    schema_version: Any = None,
    on_step: Callable[[int], None] | None = None,
    on_clamp: Callable[[], None] | None = None,
    check: Callable[[], None] | None = None,
    publisher: FramePublisher | None = None,
    preview_frame: Callable[[dict[str, Any]], None] | None = None,
    episodes_total: int | None = None,
    pace: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    episode_options: dict[str, Any] | None = None,
) -> EpisodeResult:
    """Run one (task, seed) episode. `check()` is called every step and may raise
    (lease lost / shutdown / submission budget) — such exceptions propagate after the
    journal records an `aborted` end.

    `publisher` (optional) receives the live-view frames: step 0 after /reset, one per
    /step, the last with force=True and done/success. `pace` (0 = full speed, 1.0 = real
    time at the task's control_hz) only stretches wall clock — it never touches what is
    sent to the policy/engine or journaled. Paced waits do not count against the
    episode timeout."""
    task_id = task["task_id"]
    result = EpisodeResult(ordinal=ordinal, task_id=label, seed=seed, journal_file=journal_file,
                           episode_options=episode_options)
    started = clock()
    paced_total = 0.0  # wall-clock spent in pacing waits, excluded from the episode budget
    ended = False
    hz = task.get("control_hz")
    step_period = 1.0 / hz if isinstance(hz, (int, float)) and not isinstance(hz, bool) and hz > 0 else 0.0
    pace = pace if isinstance(pace, (int, float)) and pace > 0 else 0.0
    paced = pace > 0 and step_period > 0
    episode_length = task.get("episode_length")
    policy_elapsed = 0.0
    step_cap = step_limit_steps(settings.step_limit_seconds, hz) if settings.judged else None
    image_every = settings.journal_image_every if isinstance(settings.journal_image_every, int) and settings.journal_image_every > 0 else 0

    def full_images(at_step: int) -> bool:
        return image_every > 0 and (image_every == 1 or at_step % image_every == 0)

    final_obs_state: dict[str, Any] = {"obs": None, "step": 0, "full": True}

    def emit(observation: dict[str, Any], at_step: int, final: bool, views: dict[str, Any] | None = None) -> None:
        if publisher is None:
            return
        images = observation.get("images") or {}
        frame: dict[str, Any] = {
            "top": images.get("top"), "wrist": images.get("wrist"), "task": label, "seed": seed,
            "episode": ordinal, "episodes_total": episodes_total, "step": at_step,
            "episode_length": episode_length, "done": final,
            "success": bool(result.success) if final else None,
            "goal": observation.get("goal"),
            "instruction": observation.get("instruction"),
        }
        if settings.judged:
            # A judged scenario ends when its policy-time budget is spent, not at
            # episode_length steps, so live viewers need the budget to show progress.
            frame["elapsed_seconds"] = round(max(0.0, min(policy_elapsed, settings.episode_timeout)), 2)
            frame["scenario_seconds"] = settings.episode_timeout
            if step_cap is not None:
                frame["max_steps"] = step_cap  # a fast policy reaches the step limit first
        # Audience cameras ride beside the policy observation. They are not in /act.
        if views:
            frame["views"] = views
        try:
            if preview_frame is not None:
                preview_frame(frame)
            publisher.publish(frame, force=final)
        except Exception:  # noqa: BLE001 - live view must never disturb the loop
            pass

    def _end(outcome: str) -> None:
        nonlocal ended
        ended = True
        result.digest = state_chain_digest(result.substep_hashes)
        if journal and settings.judged and image_every > 0 and not final_obs_state["full"] and final_obs_state["obs"] is not None:
            journal.write({"type": "final_observation", "step": final_obs_state["step"], "observation": final_obs_state["obs"]})
            final_obs_state["full"] = True
        if journal:
            record: dict[str, Any] = {"type": "episode_end", "finished_at": now_iso(), "outcome": outcome,
                                      "steps": result.steps, "state_chain_digest": result.digest}
            if result.failure_code:
                record["failure_code"] = result.failure_code
            if result.message:
                record["message"] = result.message[:500]
            if result.end_reason:
                record["end_reason"] = result.end_reason
            if result.held:
                record["held_steps"] = dict(result.held)
            if result.safety_peaks:
                record["safety_peaks"] = dict(result.safety_peaks)
            journal.write(record)

    try:
        options: dict[str, Any] = {'camera_configuration': settings.camera_configuration, 'max_steps': settings.max_steps}
        if settings.camera_poses is not None:
            options['camera_poses'] = settings.camera_poses
        # Judged plan options go to the engine only when they differ from its defaults, so an engine without
        # per-episode options still runs the default episodes unchanged.
        sent = engine_options(episode_options)
        if sent is not None:
            options['options'] = sent
        reset = engine.reset(task_id, seed, **options) if settings.judged else engine.reset(task_id, seed)
    except EngineInfraError:
        result.failure_code = "infra_error"
        _end("infra")
        raise
    episode_id: str = reset["episode_id"]
    result.episode_id = episode_id
    token = secrets.token_hex(16)  # per-episode runner-minted auth token (plan §4)
    obs: dict[str, Any] = reset["observation"]
    if journal and settings.judged:
        journal.write({'type': 'reset', 'response': packed(reset) if image_every > 0 else evidence_response(reset)})
    target = list(reset.get('simulator_state', {}).get('target', obs.get('proprio', {}).get('joint_pos', [])))
    target_ok = len(target) == 6 and not any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in target)
    if settings.judged and not target_ok:
        raise EngineInfraError('engine reset lacks six joint positions for motion boundary')
    # Practice (not judged) keeps the same boundary whenever the engine gave a target; without one it passes through.
    motion_on = settings.judged or target_ok
    gripper_now = reset.get('simulator_state', {}).get('gripper_target') if isinstance(reset.get('simulator_state'), dict) else None
    if isinstance(gripper_now, bool) or not isinstance(gripper_now, (int, float)):
        gripper_now = (obs.get('proprio') or {}).get('gripper')
    gripper = max(0.0, min(1.0, float(gripper_now))) if (not isinstance(gripper_now, bool) and isinstance(gripper_now, (int, float))
                                                       and math.isfinite(gripper_now)) else 1.0
    started = clock()  # reset and evidence setup are outside the scenario
    views = reset.get("views")
    step = int(reset.get("step", obs.get("step", 0)))
    done = bool(reset.get("done", False))
    if done:
        _capture_metrics(result, reset)
    last_payload: dict[str, Any] | None = None

    def fail(code: str, message: str) -> EpisodeResult:
        if last_payload is not None:
            _capture_metrics(result, last_payload)
        _team_failure(result, code, message)
        emit(obs, step, True, views)
        _end("failed")
        return result

    def finish_step_limit() -> EpisodeResult:
        """The repetition used its simulated-time cap: a normal end (no request in flight, evidence complete)."""
        if last_payload is not None:
            _capture_metrics(result, last_payload)
        else:
            _capture_metrics(result, reset)
        if result.score is None:
            result.score = 0.0
        if result.doneness is None:
            result.doneness = 0.0
        result.completed = True
        result.end_reason = STEP_LIMIT
        _end("step_limit")
        return result

    def finish_scenario_deadline(current_stepped: dict[str, Any] | None = None) -> EpisodeResult:
        if current_stepped is not None and current_stepped.get("info", {}).get("applied"):
            _capture_metrics(result, current_stepped)
        elif last_payload is not None:
            _capture_metrics(result, last_payload)
        else:
            _capture_metrics(result, reset)
        if result.score is None:
            result.score = 0.0
        if result.doneness is None:
            result.doneness = 0.0
        result.completed = True
        result.end_reason = ACT_BUDGET_EXCEEDED
        _end("scenario_time_limit")
        return result

    if journal:
        journal.write({"type": "episode_start", "submission_id": submission_id, "task_id": label, "seed": seed,
                       "episode_id": episode_id, "engine_schema_version": schema_version, "started_at": now_iso(),
                       "journal_format": JOURNAL_FORMAT, "journal_images": "none" if image_every == 0 else "all" if image_every == 1 else f"every:{image_every}",
                       "camera_configuration": settings.camera_configuration, "camera_poses": settings.camera_poses,
                       "engine": settings.engine_info,
                       **({"episode_options": episode_options} if episode_options is not None else {})})
    emit(obs, step, done, views)
    try:
        while not done and (settings.judged or not isinstance(episode_length, int) or step < episode_length):
            if check:
                check()
            if step_cap is not None and step >= step_cap:
                return finish_step_limit()
            step_started = clock()
            elapsed = policy_elapsed if settings.judged else step_started - started - paced_total
            if elapsed >= settings.episode_timeout:
                if settings.judged:
                    return finish_scenario_deadline()
                return fail("episode_budget_exceeded", f"episode exceeded {settings.episode_timeout:g}s")
            if on_step:
                on_step(step)
            body = build_envelope(obs, episode_id, step)
            request_full = full_images(step) or (image_every > 0 and step_cap is not None and step + 1 >= step_cap)
            if journal and settings.judged:
                journal.write({'type': 'policy_request', 'step': step,
                               'body': body.decode() if request_full else json.dumps(slim({**obs, 'episode_id': episode_id, 'step': step}), separators=(',', ':')),
                               'images': 'full' if request_full else 'omitted', 'episode_token': token,
                               'image_digests': image_digests(obs.get('images')),
                               'goal_digest': image_digest(obs['goal']) if isinstance(obs.get('goal'), str) else None})

            def keep_request_full() -> None:
                # A step whose /act failed or was cut keeps its full request for disputes.
                if journal and settings.judged and image_every > 0 and not request_full:
                    journal.write({'type': 'policy_request_full', 'step': step, 'body': body.decode()})
            try:
                if settings.judged:
                    act = _bounded_act(policy, body, episode_id=episode_id, step=step, token=token,
                                       budget=max(0.001, settings.episode_timeout - elapsed), check=check,
                                       record=(lambda response: journal.write({'type': 'policy_response', 'step': step, **response})) if journal else None)
                else:
                    act = policy.act(body, episode_id=episode_id, step=step, token=token)
            except _PolicyDeadline:
                keep_request_full()
                result.policy_pending = True
                return finish_scenario_deadline()
            except PolicyFailure as failure:
                keep_request_full()
                policy_duration = getattr(failure, "request_seconds", 0.0) if settings.judged else 0.0
                result.policy_seconds = policy_elapsed + policy_duration
                if journal:
                    for attempt in failure.attempts:
                        journal.write({"type": "act_failed", "step": step, "task_id": label, "seed": seed, **attempt})
                if (settings.judged and failure.code != "policy_crashed"
                        and policy_elapsed + policy_duration >= settings.episode_timeout):
                    # A known crash keeps its attribution (and the worker's OOM check). A transport timeout at
                    # the deadline is ambiguous: the request may still be in flight, so restart the policy.
                    if failure.code == "policy_timeout":
                        result.policy_pending = True
                    return finish_scenario_deadline()
                held_reason = getattr(failure, "held_reason", None)
                if held_reason is None:
                    result.team_detail = getattr(failure, "team_detail", None)
                    result.failure_reason = getattr(failure, "reason", None)
                    return fail(failure.code, failure.message)
                # A delivered but malformed response: hold position for this step and carry on.
                keep_request_full()
                policy_elapsed += policy_duration
                act_action, act_attempt, act_total_ms = None, len(failure.attempts) + 1, round(policy_duration * 1000, 1)
            else:
                held_reason = None
                if settings.judged:
                    policy_elapsed += act.request_seconds
                    result.policy_seconds = policy_elapsed
                if journal:
                    for attempt in act.failed_attempts:
                        journal.write({"type": "act_failed", "step": step, "task_id": label, "seed": seed, **attempt})
                act_action, act_attempt, act_total_ms = act.action, act.attempt, act.total_ms
                if isinstance(act_total_ms, (int, float)) and not isinstance(act_total_ms, bool):
                    result.act_ms.append(float(act_total_ms))
                if settings.judged:
                    if check:
                        check()
                    if policy_elapsed >= settings.episode_timeout:
                        return finish_scenario_deadline()
            safe_action: Any = act_action
            row_action: Any = None
            new_target = target
            element_hits: list[list[tuple[str, str]]] = []
            if held_reason is None and motion_on:
                try:
                    safe_action, new_target, clamped = clamp_action(act_action, target, hits=element_hits)
                    note_ignored(result.ignored_fields, ignored_top_level(act_action))
                    note_ignored(result.ignored_fields, getattr(act, 'ignored_fields', ()))
                    result.actions_clamped |= clamped
                    if clamped and on_clamp:
                        on_clamp()
                    # Clamped steps keep hashes and this row, but do not force full images: a policy
                    # that clamps nearly every step would otherwise journal every frame.
                    if clamped and journal and settings.judged:
                        journal.write({'type': 'clamped_action', 'step': step, 'original': act_action, 'clamped': safe_action})
                except MotionViolation as exc:
                    held_reason = exc.reason
                    keep_request_full()
            elif held_reason is None:
                note_ignored(result.ignored_fields, getattr(act, 'ignored_fields', ()))

            def note_hold(reason: str, at_step: int, original: Any, replacement: Any) -> None:
                result.held[reason] = result.held.get(reason, 0) + 1
                if journal:
                    journal.write({'type': 'held_action', 'step': at_step, 'reason': reason,
                                   'original': original, 'clamped': replacement})

            if held_reason is not None:
                safe_action = hold_action(gripper)
                new_target = target
                note_hold(held_reason, step, act_action, safe_action)
            if check:
                check()
            def step_hold(at_step: int, hold: dict[str, Any]) -> dict[str, Any]:
                """A hold-position step is always valid; the engine refusing it is an engine fault (DEC-2), not the team's."""
                try:
                    return engine.step(episode_id, at_step, hold)
                except ActionRejected as exc:
                    raise EngineInfraError(f"engine rejected a hold-position step: {exc}") from exc

            if not motion_on:  # practice without a known joint target: the action is passed through unvalidated
                try:
                    stepped = step_hold(step, safe_action) if held_reason is not None else engine.step(episode_id, step, safe_action)
                except ActionRejected:
                    safe_action = hold_action(gripper)
                    note_hold('engine_rejected', step, act_action, safe_action)
                    stepped = step_hold(step, safe_action)
            else:
                # DEC-2, judged and practice alike: chunk elements go to the engine one at a time. Engine steps are
                # irreversible, so elements already applied stay applied; the element the engine rejects and the rest
                # of the chunk are replaced by ONE hold-position step.
                actions = safe_action['actions'] if safe_action.get('type') == 'action_chunk' else [safe_action]
                hashes_all: list[str] = []
                applied = 0
                stepped = None
                position = list(target)
                synced = False
                chunk_gripper = gripper
                sent: list[dict[str, Any]] = []  # what the engine really received, for the journal and replay
                for index, action in enumerate(actions):
                    if check:
                        check()
                    engine_held = False
                    try:
                        response = (step_hold(step + applied, action) if held_reason is not None
                                    else engine.step(episode_id, step + applied, action))
                    except ActionRejected:
                        action = hold_action(chunk_gripper)
                        note_hold('engine_rejected', step + applied, act_action, action)
                        engine_held = True
                        new_target = position
                        response = step_hold(step + applied, action)
                    sent.append(action)
                    if not engine_held:
                        engine_target = _engine_target(response)
                        synced = synced or engine_target is not None
                        position = engine_target or [p + d for p, d in zip(position, action['values'])]
                        chunk_gripper = action['gripper']
                    if journal and settings.judged:  # practice journals keep their one row per step
                        response_full = image_every > 0 and (
                            full_images(step + applied) or bool(response.get('done'))
                            or (step_cap is not None and step + applied + 1 >= step_cap))
                        # Default: digests + packed simulator state only. Image bodies only on request (RC_JOURNAL_IMAGES).
                        journal.write({'type': 'engine_response', 'step': step + applied,
                                       'response': packed(response) if response_full else evidence_response(response)})
                        if image_every > 0:
                            final_obs_state.update(obs=response.get('observation'), step=step + applied + 1, full=response_full)
                    info = response['info']
                    hashes_all.extend(info.get('substep_hashes', []))
                    if (held_reason is None and not engine_held and int(info.get('applied', 0)) > 0
                            and index < len(element_hits)):
                        add_clamps(result.clamps, element_hits[index])
                    _note_safety(result, info)
                    if add_speed_scaled(result.clamps, info):
                        result.actions_clamped = True
                        if on_clamp:
                            on_clamp()
                    applied += int(info.get('applied', 0))
                    stepped = response
                    if response['done'] or info.get('deadline_expired') or not info.get('applied') or engine_held:
                        break
                if stepped is None:
                    return finish_scenario_deadline()
                stepped = {**stepped, 'info': {**stepped['info'], 'applied': applied, 'substep_hashes': hashes_all}}
                target = position if synced else new_target
                gripper = chunk_gripper
                if any(a is not b for a, b in zip(sent, actions)) or len(sent) != len(actions):
                    row_action = {'type': 'action_chunk', 'actions': sent}  # an engine rejection cut the chunk short
            info = stepped["info"]
            if check:
                check()
            hashes = [h for h in info.get("substep_hashes", []) if isinstance(h, str)]
            result.substep_hashes.extend(hashes)
            if journal:
                journal.write({
                    "type": "step", "ts": now_iso(), "task_id": label, "seed": seed, "episode_id": episode_id,
                    "step": step, "action": row_action if row_action is not None else safe_action if held_reason is not None else act_action, "applied": info.get("applied"),
                    "state_hash": info.get("state_hash"), "substep_hashes": hashes,
                    "done": bool(stepped["done"]), "success": bool(stepped.get("success", False)),
                    "attempt": act_attempt, "act_ms": {"total": act_total_ms},
                    "obs_bytes": len(body), "obs_sha256": hashlib.sha256(body).hexdigest(),
                })
            obs, done = stepped["observation"], bool(stepped["done"])
            views = stepped.get("views")
            step = int(stepped.get("step", step + 1))
            result.steps = step
            result.success = bool(stepped.get("success", False))
            last_payload = stepped
            emit(obs, step, done or (not settings.judged and isinstance(episode_length, int) and step >= episode_length), views)
            if paced:  # sleep the remainder of this step's wall-clock budget
                wait_start = clock()
                _wait(step_period * pace - (wait_start - step_started), check, clock, sleep)
                paced_total += clock() - wait_start
        result.completed = True
        if last_payload is not None:
            _capture_metrics(result, last_payload)
        _end("success" if result.success else "failed")
        if paced and settings.final_hold > 0:  # let a viewer see the result; abortable, after the journal is closed out
            _wait(settings.final_hold, check, clock, sleep)
        return result
    except EngineInfraError as exc:
        result.failure_code, result.message = "infra_error", str(exc)
        _end("infra")
        raise
    except BaseException:
        if not ended:
            result.message = "aborted"
            _end("aborted")
        raise
