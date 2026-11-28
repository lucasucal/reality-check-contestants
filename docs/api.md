# Policy API

Your policy is an HTTP server. The runner sends `POST /act` for each control
step. There is no `/episode_start` or `/episode_end`; the runner stops calling
when the scenario ends. Return a valid action for the same `(episode_id, step)`
every time.

## Startup routes

| Route | Contract |
|---|---|
| `GET /healthz` | `200 {}` as soon as the process starts. Liveness only; do not wait for model loading. |
| `GET /readyz` | `200` with `ready`, `contract: "v0"`, `schema_version: 3`, and capabilities. `capabilities` (`chunking`, `caching`) is informational: the runner never reads it and always accepts an `action_chunk` of 1 to 16 actions. Set ready only after weights and model load. A terminal load failure returns `{ "ready": false, "error": "…" }` with a bounded generic message; print full details to stdout or stderr (the container log, shown to the team after the run up to the hidden tasks). The runtime may add an optional short `error_code` (for example `entry_import_failed`, `entry_not_policy`, `load_failed`). |
| `POST /act` | Accept an observation envelope and return an action. Echo `episode_id` and `step` in the JSON response and `X-RC-Episode-Id` / `X-RC-Step` headers. |

The judged runner mounts the pinned Hugging Face snapshot read-only at `/weights`
([model checks](#model-checks-at-submit-and-at-run-time)). The container always serves on 8080. Who answers the three
routes depends on the manifest `entry_point` (see [SUBMIT.md](../contestants/SUBMIT.md)):
for a `package.module:Attr` Policy class, the image's `rc_runtime` serves them for you;
for a program (a dotted module or a relative `.py` path), your own program serves them,
listening on `0.0.0.0:$RC_POLICY_PORT` (8080), and everything on this page (readiness
semantics, echo rule, action and chunk rules) is your program's responsibility. A
program that exits fails the run, and so does one that has not started listening on the port
within 10 minutes (`RC_POLICY_LISTEN_DEADLINE_SECONDS`, default 600; the runtime exits with
code 78; a server bound only to 127.0.0.1 does not count): start the server first, load slow models afterwards and report not-ready via `/readyz`. A
missing or empty mount must fail loading; policy code must not silently fetch
or substitute another model. A local development fallback is permitted only
when explicitly enabled by the policy developer.

Class policies declare `"schema_version": 3` in `manifest.json`. A program
advertises v3 directly in `/readyz`. The runner checks the version before
the first episode. Migrate policies written for schema v2 by updating their
input handling and checking their output under the v3 action limit. Verify
the selected checkpoint's camera features, preprocessing and action units,
then re-evaluate or retrain as needed.

### Whistle mock deployment

On Spark, start the policy independently and keep it on the same private
container network as the Whistle mock runner. Give the policy a stable DNS
service name and configure Whistle with an HTTP origin such as
`--policy http://whistle-policy:8080`. The runner sends the v3 request
directly to the policy.

## Request

`POST /act` takes a JSON object with all fields below. Each image is a
640×360 RGB JPEG encoded with standard base64. Both `images` and `goal` carry
all three named views. `/tasks` publishes their dimensions in `obs_spec`.

| Configuration | Image keys | Meaning |
|---|---|---|
| `three_view` | `wrist`, `left`, `right` | Wrist-mounted, left-side, and right-side views |

The three `goal` images depict this episode's finished scene with the same
parts, variant, fixture pose and lighting. They stay fixed for the repetition
and arrive on every request. A policy may ignore them. Judging uses
the `three_view` configuration and the
engine's default camera poses, the ones practice uses (`cell.json` in
`engine/rc_engine/assets/cell`; no overrides). The wrist camera is fixed to the
bracket. Each episode jitters the two side cameras about their nominal poses, within the
public ranges by up to ±1.5 cm along each axis and ±1.5° about each axis; some judged
repetitions vary the scene more than the public ranges, so do not hard-code side-camera
extrinsics to the pixel ([scene variation](../contestants/RULES.md#scene-variation)).
The engine renders without multisampling (`RC_RENDER_OFFSAMPLES=0`), so the same seed
and the same actions give identical pixels on every run, in practice and in judging.
`instruction` is constant within an episode but is one phrasing drawn per
episode from the task's pool ([instructions](../contestants/RULES.md#instructions)).

```json
{
  "schema_version": 3,
  "episode_id": "8f3a",
  "step": 0,
  "task_id": "bolt_to_tray",
  "instruction": "Pick up the bolt and place it in the tray.",
  "images": {
    "wrist": "<base64 JPEG>",
    "left": "<base64 JPEG>",
    "right": "<base64 JPEG>"
  },
  "goal": {
    "wrist": "<base64 JPEG>",
    "left": "<base64 JPEG>",
    "right": "<base64 JPEG>"
  },
  "proprio": {
    "joint_pos": [-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708],
    "joint_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gripper": 0.9968,
    "ee_pos": [0.8906, -0.0078, 0.2016],
    "ee_quat": [-0.0289, -0.6302, 0.7751, 0.0355]
  }
}
```

`joint_pos` and `joint_vel` use the order shoulder pan, shoulder lift, elbow,
wrist 1, wrist 2, wrist 3. Joint positions are radians and velocities are
rad/s. `ee_pos` gives the palm position in metres and `ee_quat` gives its unit quaternion in
wxyz order. Both are relative to UR `base_link`, with its origin at the base centre on the mounting
plane and z up. The tabletop is at z = −0.187 in this frame. The palm is the simulator's
`gripperframe` site, with +Z pointing in the approach direction and +Y along finger opening;
the real runner's native palm uses a different frame and requires an explicit conversion before
comparing these fields.
Measured gripper opening is normalized from 0 (closed) to 1 (open).
`task_id` is a public task id or an anonymous hidden id (`"1"` to
`"4"`). Hidden task names and per-repetition outcomes are not exposed; after
scoring, each hidden task's score is shown under that anonymous id.

## Response and action limits

The response body wraps the action and echoes the request identifiers. Return
the same values in the `X-RC-Episode-Id` and `X-RC-Step` HTTP headers.

```json
{
  "episode_id": "8f3a",
  "step": 0,
  "action": {
    "type": "joint_delta",
    "values": [-0.004, 0, 0.01, 0, 0, 0],
    "gripper": 1.0
  }
}
```

An action is either one `joint_delta` or an `action_chunk` containing 1–16
`joint_delta` actions. `chunk_len`, if present, must match the list length:

```json
{"type": "action_chunk", "actions": [
  {"type": "joint_delta", "values": [0, 0, 0, 0, 0, 0], "gripper": 1.0},
  {"type": "joint_delta", "values": [0, -0.02, 0.02, 0, 0, 0], "gripper": 1.0}]}
```

A `Policy.act()` returns the `action` object. Each `joint_delta` uses only
`type`, `values` and `gripper`. A chunk uses `type`, `actions` and optional
`chunk_len`. The runner ignores extra fields at the response or chunk level.
Mixing `values` or `gripper` into an `action_chunk`, or putting `actions` or
`chunk_len` in a `joint_delta`, makes the response malformed.
Each action has six finite joint increments and a finite, absolute gripper
target. The policy boundary clamps each increment to ±0.05 rad per 50 ms
step and the gripper target to `[0, 1]`. It validates the complete chunk
before applying any element and clips a valid chunk to the remaining step budget. Joint target limits are ±6.28319 radians except elbow
±2.792526803190927 radians (±160°). Finite out-of-limit increments, gripper
values and joint targets are clamped to the legal limits before stepping, and
the team sees a notice. The simulator then scales the whole step down (one
factor for all six increments, direction kept) when it would move the TCP
(`gripperframe` site) or the elbow (`forearm_link` origin) faster than 1.5 m/s,
computed as |J(q_target) · Δq| / 0.05 s with the MuJoCo position Jacobian at the
commanded joint target; those steps are counted as "TCP speed" / "elbow speed"
in the same notice. TCP force (150 N), momentum (25 kg·m/s) and mechanical power
(300 W) are measured, never enforced or scored. When a public repetition exceeds
one of these thresholds, the result includes a notice based on the UR12e's
default (Normal) safety limits
([RULES](../contestants/RULES.md#action-boundary)). Acceleration and Cartesian
workspace limits are not enforced, and none of this is a real robot safety
system.
For a malformed response, including an incorrect body or header echo, the
runner applies one zero-delta step with the previous gripper target. It
discards every element of an invalid chunk. The result page counts held steps.
`ee_delta` is unsupported. A decoded response body larger than 2 MiB ends that
repetition (`response_too_large`, scored as it stands); responses are small JSON
actions, so do not add large extra fields.

### How actions move the arm

The arm follows a commanded joint target. The observation does not include it,
but a policy can track it exactly:

- At step 0 the target is the home pose, `HOME_Q` =
  `[-0.2055, -1.1091, 1.6007, -1.9708, -1.5708, 1.5708]` rad (shoulder pan,
  shoulder lift, elbow, wrist 1, wrist 2, wrist 3).
- Each applied `joint_delta` adds its `values` to the target, not to
  `joint_pos`. The values are clamped first: each increment to its joint's
  per-step limit (0.05 rad), then the sum to the joint target
  limits above (±6.28319 rad, elbow ±2.792526803190927 rad). The added value is
  the clamped one, scaled down when the step went over the 1.5 m/s TCP or elbow
  speed limit.
- A held step adds nothing (its `values` are zeros).
- The elements of an `action_chunk` apply in order, one 20 Hz step each, and
  each one adds to the target the previous one left.
- `gripper` is not summed: each action sets the gripper target directly.
- The servo moves the commanded position toward the target with a 0.1 s
  lookahead, so `joint_pos` lags the target while the arm moves. Contact can
  also keep the arm off it.

To track the target, start from `HOME_Q` at step 0 and add each applied
increment after clamping it the same way
([action boundary](../contestants/RULES.md#action-boundary)). Add the next
action to that sum, not to `joint_pos`. The sum is exact as long as your steps
stay under 1.5 m/s at the TCP and the elbow; a scaled step adds less.

Schema v3 runs make one attempt per `/act` call. Each repetition has a 90-second
cumulative budget for elapsed `/act` HTTP requests, including connection and
transport time. Runner processing and simulator work do not count. Each call's
absolute deadline is capped by the remaining request budget; the connect
timeout is at most five seconds and may be shorter. Judged
loading and warmup have a 30-minute deadline (1800 s per policy container
start; the runner restarts the policy before each task by default); organizers
can also stop a submission manually. All limits are in the
[limits table](../contestants/RULES.md#limits). Startup readiness failures should return a bounded,
generic error while preserving diagnostic detail in container logs.

### `/act` budget arithmetic

A repetition is at most 1,200 control steps. A policy that returns 1 to 4 actions per call
makes 1,200 to 300 calls per repetition, so the 90 s budget allows 75 to 300 ms per call on
average for everything: transport, the JPEG-encoded images in every request, decoding and inference.

| Actions per call | Calls per repetition | Budget per call | Fixed overhead at 23 ms per call |
|---|---|---|---|
| 1 | 1,200 | 75 ms | 27.6 s |
| 4 | 300 | 300 ms | 6.9 s |
| 16 (`action_chunk` maximum) | 75 | 1,200 ms | 1.7 s |

The 23 ms per call is provisional: a practice team measured it as the `/act` p50 of calls
that skipped perception, in the judged image on their own harness. The
budget stays 90 s. To cut the call count, return
an `action_chunk` (up to 16 actions per call); `practice` prints your `/act` seconds per
repetition against the budget.

## Robot model notes for your own IK

The robot model is in `engine/rc_engine/embodiment.py`. The policy image ships the same
MuJoCo version as the judged simulator (4.0.1), so loading that model in your policy gives
the simulator's kinematics.

- The arm collides through primitives, not its visual meshes (`_apply_arm_collisions`):
  capsules on the shoulder, upper arm, forearm, wrist 1 and wrist 2, and a cylinder 90 mm
  across and 50 mm long on wrist 3. Check clearance against these.
- On the default IK branch, holding the tool horizontal about 15 cm above the table (for
  example a bolt head-up over a hole) puts the wrist 1 capsule into the table. Choose
  another branch or heading and check the capsule heights.
- At the home pose the parked gripper can hide parts from the left camera, and the wrist
  camera sees parts in the spawn area only cut off at the bottom edge, behind the fingers.

## Judged run shape

Six tasks (two public, four hidden), each run four times from unpublished starts that are the same
for every team and submission: 24 repetitions inside a 90-minute (5400 s) run deadline. A repetition
ends on success, when its 90 s `/act` budget runs out, or after 60 simulated seconds
(`step_limit_seconds`, 1,200 control steps; `end_reason: "step_limit"`, not a failure, scored on the
state reached). The opt-in protective stop (`RC_SAFETY_STOP=enforce`, off in judged runs unless
organizers announce it) ends a repetition on the first 50 ms step over a UR12e force, momentum or power
limit (`end_reason: "protective_stop"`, scored on the state reached). The speed bonus counts control
steps, not `/act` time. Rules, scoring and the scene variation: [RULES](../contestants/RULES.md). A
submission accepted before the cutoff counts even if judging finishes after it. Known issues in the
judged image: [KNOWN-ISSUES](../contestants/KNOWN-ISSUES.md).

## Leaderboard ranks

`GET /api/leaderboard` returns `[{"rank", "score"}]` and `GET /api/rank` returns
`{"rank", "score", "position"}` for the caller's team (`null`s when unranked; `position` is the
0-based index of the team's own row in `/api/leaderboard`). One ranking for all teams, by each team's
best score, rounded half up to 2 decimals ([leaderboard](../contestants/RULES.md#leaderboard)).

- Rows with the same rank are listed in a stable order; team names are not in the payload.
- A submission the organizers excluded from the leaderboard never counts; the team's next best
  scored run takes its place. That submission's `GET /api/submissions/{id}` then carries
  `leaderboard_exclusion: {"category", "label", "text"}`.

## Model checks at submit and at run time

What is checked when: [your model](../contestants/SUBMIT.md#your-model).

- `POST /api/submissions` checks the model by Hugging Face metadata only and returns `201` at once; the
  hourly cooldown starts on acceptance. A refused model does not start the cooldown; a Hugging Face
  outage during the check returns `503` with `retryable: true`.
- Run-time model failures: `weights_unavailable` (repo deleted, private or gated, or pinned commit
  gone; charged), `weights_unsafe_entries`, `weights_check_failed`, `weights_snapshot_too_large`
  (charged), `weights_verify_failed` and `weights_provider_unavailable` (not charged; the run is
  retried after a provider failure). `failure.stage` is `prepare`.

## Submission status for teams

`GET /api/submissions/{id}` (and the list) carries these fields besides state,
pins and score. Public tasks always run before hidden tasks; apart from
`hidden_task_scores`, nothing below describes a hidden task.

| Field | When | Content |
|---|---|---|
| `failure` | `failed` | `stage`, `code`, `reason`, `title`, `explanation`, `detail`, `charged`, `retryable` (resubmit now; equals `!charged`), `next_steps[]`, `docs[{label, url}]`, `log`, `at`, and `known_issue` when matched. For `stage: "hidden_evaluation"`, `reason` and `detail` are `null`, `known_issue` is absent and the explanation is the general one for the code. |
| `evaluation_summary` | `scored`, `failed` | `public`: one row per public repetition (`task`, `repetition`, `success`, `steps`, `end_reason`, `failure_code`, `actions_clamped`, `score`, `quality`, `quality_limit`). `score` is the repetition's points (2 decimals, every repetition including failed ones; `null` in rows from older runners); `quality` is the finish precision 0..1 and `quality_limit` the measurement that limited it, both for successful repetitions only (else `null`). `hidden`: `abnormal_endings`, true if any hidden repetition ended because the policy crashed, timed out or returned an error, and (DEC-6) `ended_early`: `{policy_crashed, policy_timeout, policy_error, act_budget_exceeded}` counts summed over every hidden task and repetition (closed keys, always all present; absent in summaries from older runners). No task, repetition, step, message or timing. `notes`: act-budget, step-limit and clamping counts over public repetitions only; `notes.clamped` = `{episodes, of, steps_of, limits}` where `steps_of` is the applied steps of the public repetitions and `limits` is `{limit: {joint: steps}}` with `limit` in `step_increment` (any joint), `joint_limit` (any joint but `elbow`), `elbow_range` (`elbow`), `gripper_range` (`gripper`), `tcp_speed` (`tcp`) and `elbow_speed` (`elbow`; steps the simulator scaled to 1.5 m/s); joints `shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`. `notes.safety_stop` (only when it happened) = `{of, tcp_force, momentum, power}`: public repetitions whose peak went over 150 N, 25 kg·m/s and 300 W, the UR12e's default safety limits (they would trigger a safety stop on the real arm; not scored, and enforced only by the opt-in protective stop). `notes.protective_stop` (only with that stop, when it ended a public repetition) = `{count, of}`; the run notice is then one `safety_stop` notice saying how many repetitions the simulator ended. `end_reason` is one of `done, act_budget_exceeded, step_limit, protective_stop, policy_error, infra_error, incomplete`. `resources` (public phase only; absent from older runners): `load_seconds` (container start to `/readyz` ready, one per policy start before the hidden tasks, at most 16), `memory_peak_gib` (combined CPU + GPU peak measured by the memory watchdog up to the first hidden `/act`, 0.1 GiB) and `act_ms` `{calls, p50, p95}` over the public repetitions' `/act` calls; `null` when not measured. |
| `score_total`, `public_task_scores`, `hidden_task_scores` | `scored` | `public_task_scores` is `{public task id: score}` (public task ids only); `hidden_task_scores` is `{anonymous hidden id: score}` with the ids sent to the policy (`"1"` to `"4"`), each the mean of that task's four repetitions ([evaluation privacy](../contestants/RULES.md#leaderboard)). A judged run (hidden tasks scored) rounds all three half up to the leaderboard's two decimals; a public-only run keeps the exact total and public task scores and has `hidden_task_scores` `{}`. Same values as `/score` `total`, `public_tasks` and `hidden_tasks` (`display.public_tasks` and `display.hidden_tasks` are formatted like `display.score_text`); `null` before. Scores from older runners carry no hidden task scores. |
| `progress` | `building`, `running` | `phase`, `phase_started_at`, and in the public part `load` (`container_start`, `readyz`, `error_code`) and `public` (`done`, `total`, `episodes`). During `hidden_evaluation` only `phase` and `phase_started_at`. |
| `run_started_at`, `deadline_at`, `load_deadline_s` | `building`, `running` | Run start, total run deadline, load deadline per policy start in seconds. |
| `last_heartbeat_age_s` | `building`, `running` | Seconds since the machine last reported, in 15 s steps. `updated_at` is also in 15 s steps. |
| `requeue_notice` | after a requeue | `text`, `attempt`, `max_attempts`, `at`: the machine stopped responding and the run restarted elsewhere; not charged. |
| `log` | `scored`, `failed` | `available`, `phases` (from `start`, `warmup`, `public_evaluation`), `withheld` (`"hidden_evaluation"` when hidden-task output was left out). The runtime's own successful request lines (`[rc_runtime] <ip> "POST /act HTTP/1.1" 200 -`, likewise `GET /readyz`, `GET /healthz`, and the `[rc_sdk]` server's) are left out of each snapshot before the 50 + 150 line cap, replaced by one `... N runtime request lines omitted ...` line; everything your policy prints stays. |

Stages: `prepare`, `queue`, `start`, `warmup`, `public_evaluation`,
`hidden_evaluation`, `scoring`, `recording`, `control`. Codes, reasons and the
charged rule: [SUBMIT.md](../contestants/SUBMIT.md#what-you-see-before-during-and-after-the-hidden-tasks).
`docs[].url` links to the same guides on the dashboard (`/docs/SUBMIT.md#...`).

`GET /api/submissions/{id}/log` returns the policy container output from before
the hidden tasks:

```json
{"submission_id": 1,
 "snapshots": [{"label": "start_1", "phase": "start", "text": "..."},
               {"label": "public_1", "phase": "public_evaluation", "text": "..."}],
 "hidden_withheld": true,
 "withheld_notice": "Output from the hidden evaluation is not shown."}
```

One snapshot per policy start (label `start_<k>` or `public_<k>`), cut before that
container's first hidden request; containers started during the hidden tasks
contribute none. 404 unless the submission is `scored` or `failed` and a log was
stored. Another team's submission id answers exactly like an unknown id (404
`submission not found`) on every `/api/submissions/{id}` route, so ids cannot be enumerated. Size limits are in the
[limits table](../contestants/RULES.md#limits).
