# Reality Check: guide for coding agents

You are helping a team build a policy for Reality Check, a robotics hackathon. The policy drives a
simulated UR12e arm with a Robotiq 2F-140 gripper through fastener tasks. It is scored automatically in
simulation, and the best main-track teams run it on the real arm. This file sits at the root of the
participant bundle (`reality-check-practice/`). Paths below are relative to it.

## Read in this order

| File | What it tells you |
|---|---|
| `contestants/README.md` | The event at a glance |
| `contestants/RULES.md` | Tracks, scoring, tasks, limits |
| `docs/api.md` | The policy HTTP contract: observations, actions, timing |
| `contestants/SUBMIT.md` | How to package code and weights so the judged run loads them |
| `starter/README.md` | The template the team's policy repository starts from |
| `contestants/PRACTICE.md` | Running the policy locally in the judged image on the public tasks |
| `contestants/KNOWN-ISSUES.md` | Known failures and their fixes |

## What a submission is

- **Code:** a public GitHub repository, started from a copy of `starter/`. `manifest.json` names the
  entry point: a `Policy` class (`"policy.example:ExamplePolicy"`) or a program serving HTTP on
  `0.0.0.0:8080`.
- **Weights:** one public, ungated Hugging Face model repository containing everything the policy loads.
  It is mounted read-only at `/weights` (`RC_WEIGHTS_DIR`). Never put weights in GitHub.
- **Team:** exactly 4 people including the lead, formed in EngineExchange; the team can submit once
  all 4 are approved and checked in. A submission counts for the team the submitter is on at that moment.
- **Track:** optional for now (undeclared counts as Open): Model Training (a trained model drives the
  arm; add a short method note) or Open (any approach).
- **Cadence:** one accepted submission per team per hour; the best score counts. The 4 hidden tasks
  (of 6) carry 75% of the total.

## Constraints the code must respect

- Python only, using the packages in the published image plus the repository's own code. Nothing is
  installed at judging time.
- Offline: no network. Load everything from `/weights` (or the curated base cache, if published).
- `/scratch` is the only writable path.
- Every `/act` response echoes `episode_id` and `step`, and returns `joint_delta` (6 values plus a
  gripper target in [0, 1]) or an `action_chunk` of up to 16 actions. Out-of-range values are clamped.
- A repetition lasts at most 60 simulated seconds (1,200 steps at 20 Hz), and the policy's `/act`
  response time is capped at 90 s cumulative per repetition.
- Read the instruction and the goal image on every step. The judged starting scenes are not published.

## Commands

```sh
# In the team's policy repository (a copy of starter/)
python3 scripts/selftest.py --weights /path/to/snapshot   # must end with ALL PASS
# "CONTRACT ONLY: ..." means your load()/act() did not run here; it says why. "FAILED: ..." names the failed check

# From the bundle root: run the policy in the judged image on the public tasks
(cd runner && uv sync --group practice)
runner/.venv/bin/python -m practice run --repo /path/to/policy --model https://huggingface.co/<org>/<model>
```

`contestants/PRACTICE.md` covers starting the public engine and every option.

## Before you change anything

Run the self-test, read `docs/api.md` end to end, and check the limits table in `contestants/RULES.md`.
When a run fails, the dashboard shows an error code with a title and next steps. Look the code up in
`contestants/SUBMIT.md` and `contestants/KNOWN-ISSUES.md` before guessing.
