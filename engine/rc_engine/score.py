"""Locked episode scores in points: completeness, finish quality and time. Pure functions: no scene and no clock.

A task has n goals (stages or placements, n >= 1). Per repetition, with T the control-step budget:

- Completed (the task's success test passed, including the hold):
      score = COMPLETION_POINTS + Q * (QUALITY_POINTS + B)                 (70 .. no upper cap)
      B = SPEED_POINTS * ln(T / t) / ln(SPEED_RATIO)                       (B >= 0; 0 at t = T)
  t is the control steps used through the end of the success hold (t >= 1), not wall-clock time: every
  SPEED_RATIO-fold faster finish adds SPEED_POINTS (+10 points every time you halve your time). Q in [0, 1] is the finish quality: every required
  measurement of every goal part (position and, where the task requires it, orientation error against the
  intended seated pose; symmetric parts against the nearest acceptable orientation) scores
  measurement_quality(err, pass): 1 inside the ideal band (QUALITY_IDEAL_FRACTION of the pass tolerance),
  falling linearly to 0 at the pass tolerance the success test uses. Q is the minimum over the
  measurements, the goal parts and the samples of the success hold. So a completion scores 70 at Q = 0
  whatever its time. QUALITY_IDEAL_FRACTION is provisional, to be calibrated on real finishes.
- Not completed:
      score = INCOMPLETE_POINTS / STAGE_CREDIT * (STAGE_CREDIT * k / n + PROGRESS_WEIGHT * (1 / n) * sum(progress_i))
  i.e. 15/70 of 100 times the engine-v1 incomplete score, always below INCOMPLETE_POINTS (15).
  k is the number of goals met and counted (placed, released, at rest, not in a wrong seat; ordered tasks
  count only goals met in order; any decoy violation makes k = 0 and every progress_i = 0). progress_i is
  the goal's doneness point (fraction of its starting distance closed beyond a 1 cm dead zone, 1 inside the seat region, 0 when the
  part sits in a wrong seat), summed over the goals not counted. No speed or quality component. If every goal
  is met but the success hold was not reached, the last goal counts as progress 1 instead of a stage, so a
  non-completion stays below INCOMPLETE_POINTS and any completion beats any non-completion.

Energy (mechanical work) is not part of the score (organizer decision, 2026-11-22).
A task's score is the mean over its repetitions; the contest total is
(p1 + p2 + 1.5 * (h1 + h2 + h3 + h4)) / 8, computed by the runner from the tasks run.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

SCORER = "engine-v2"

COMPLETION_POINTS = 70.0
QUALITY_POINTS = 5.0
SPEED_POINTS = 10.0
SPEED_RATIO = 2.0
INCOMPLETE_POINTS = 15.0
STAGE_CREDIT = 0.70
PROGRESS_WEIGHT = 0.20
# Ideal band of a quality measurement, as a fraction of its pass tolerance. Provisional (to be calibrated).
QUALITY_IDEAL_FRACTION = 0.25
# Distance (metres) a part must close beyond its starting distance before it earns any progress. Parts keep
# settling by well under a millimetre after the reset settle; without this, a policy that never touches a part
# earns progress from that drift.
PROGRESS_DEADZONE_M = 0.01


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def speed_bonus(steps_used: int, step_budget: int) -> float:
    """B = SPEED_POINTS * ln(T / t) / ln(SPEED_RATIO), t = max(steps_used, 1); 0 at or past the budget."""
    if step_budget <= 0:
        return 0.0
    t = max(int(steps_used), 1)
    return max(0.0, SPEED_POINTS * math.log(step_budget / t) / math.log(SPEED_RATIO))


def measurement_quality(err: float, pass_tol: float, ideal_fraction: float = QUALITY_IDEAL_FRACTION) -> float:
    """1 for err <= ideal (ideal_fraction * pass_tol), 0 for err >= pass_tol, linear in between."""
    ideal = ideal_fraction * pass_tol
    if err <= ideal:
        return 1.0
    if err >= pass_tol:
        return 0.0
    return (pass_tol - err) / (pass_tol - ideal)


def doneness_point(distance_now: float, distance_start: float, in_region: bool) -> float:
    """One goal point against the placement target the task already uses.

    Inside that success region the point is 1. Otherwise it is the fraction of
    the starting distance closed beyond the first PROGRESS_DEADZONE_M, clamped
    to [0, 1]: (start - now - deadzone) / (start - deadzone).
    """
    if in_region:
        return 1.0
    reach = distance_start - PROGRESS_DEADZONE_M
    if reach <= 0.0:
        return 0.0
    return _clamp((distance_start - distance_now - PROGRESS_DEADZONE_M) / reach, 0.0, 1.0)


def completed_score(bonus: float, quality: float) -> float:
    """A completed repetition: COMPLETION_POINTS + Q * (QUALITY_POINTS + B), Q clamped to [0, 1], B to >= 0."""
    return COMPLETION_POINTS + _clamp(quality, 0.0, 1.0) * (QUALITY_POINTS + max(bonus, 0.0))


def incomplete_score(counted: int, unmet_progress: Sequence[float], goals_total: int) -> float:
    """A repetition that did not complete the task, in points (below INCOMPLETE_POINTS).

    `counted` goals weigh STAGE_CREDIT / n each; every goal not counted adds PROGRESS_WEIGHT / n times its
    progress point (clamped to [0, 1]); the sum is scaled by INCOMPLETE_POINTS / STAGE_CREDIT. Callers pass
    counted = 0 and no progress after a decoy violation.
    """
    if goals_total <= 0:
        return 0.0
    k = min(max(counted, 0), goals_total)
    progress = [_clamp(float(p), 0.0, 1.0) for p in unmet_progress]
    if k == goals_total:  # every goal met, success hold not reached: the last one is progress, not a stage
        k, progress = goals_total - 1, [1.0]
    fraction = (STAGE_CREDIT * k + PROGRESS_WEIGHT * sum(progress)) / goals_total
    return INCOMPLETE_POINTS / STAGE_CREDIT * fraction


def episode_score(success: bool, bonus: float = 0.0, quality: float = 1.0, counted: int = 0,
                  unmet_progress: Sequence[float] = (), goals_total: int = 1) -> float:
    """The per-repetition score: completed_score on success, incomplete_score otherwise."""
    if success:
        return completed_score(bonus, quality)
    return incomplete_score(counted, unmet_progress, goals_total)
