# Reality Check participant guide

Train a policy that drives a simulated UR12e arm through fastener tasks. Submit a public GitHub
repository (your Python code) and a public Hugging Face model (your weights) in EngineExchange.
Judging runs your code in the published policy image on an NVIDIA H100 (amd64).

## Start here

1. Read the [rules](RULES.md) and the [policy API](../docs/api.md).
2. Copy the starter template ([`starter/`](../starter)) and run `python3 scripts/selftest.py`.
3. Practice on the public tasks in the judged image: [PRACTICE.md](PRACTICE.md).
4. Package code and weights and submit: [SUBMIT.md](SUBMIT.md).
5. Hit an error? Check [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

## The event at a glance

| | |
|---|---|
| Team | Exactly 4 people including the lead, formed in EngineExchange; all 4 approved and checked in to submit; locked at the cutoff ([teams](RULES.md#teams)) |
| Tasks | 6: 2 public (`bolt_to_tray`, `bolt_and_nut_to_fixture`) and 4 hidden (75% of the total) |
| Judged run | 4 repetitions per task, from unpublished starts that are the same for every team |
| Per repetition | 60 simulated seconds (1,200 steps at 20 Hz) and 90 s of cumulative `/act` time |
| Policy input | Instruction, goal image, three 640×360 RGB cameras (`wrist`, `left`, `right`), joint state |
| Policy output | Joint deltas plus a gripper target, one action or a chunk of up to 16 |
| Score | Finish, then precision, then speed: completed = 70 + precision × (5 + speed bonus), not completed under 15 ([scoring](RULES.md#scoring)) |
| Tracks | Main track **Model Training** (a trained model drives the arm; top 3 go to the real-arm final) and side track **Open** (any solution; the judged winner gets an unranked real-arm demo). Declaring a track is optional for now; undeclared counts as Open |
| Submissions | One accepted per team per hour; your best score counts |

Every number is in the [limits table](RULES.md#limits). Cutoff and results times are in the dashboard.
