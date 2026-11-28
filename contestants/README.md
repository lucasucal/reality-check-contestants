# Reality Check participant guide

Train a policy that drives a simulated UR12e arm through fastener tasks. Submit a public GitHub
repository (your Python code) and your own public, ungated Hugging Face model repository (your weights)
in EngineExchange. Gated models cannot be submitted; upload the weights even if the pretrained model is
unchanged. See [your model](SUBMIT.md#your-model).
Judging runs your code in the published policy image on an NVIDIA RTX PRO 6000 (amd64).

## Start here

1. Read the [rules](RULES.md) and the [policy API](../docs/api.md).
2. Copy the starter template ([`starter/`](../starter)) and run `python3 scripts/selftest.py`.
3. Practice on the public tasks in the judged image: [PRACTICE.md](PRACTICE.md).
4. Package code and weights and submit: [SUBMIT.md](SUBMIT.md).
5. Hit an error? Check [what's in the image](KNOWN-ISSUES.md).

## The event at a glance

| | |
|---|---|
| Team | 3 or 4 people including the lead, formed in EngineExchange; every member must be approved and checked in to submit; locked at the cutoff ([teams](RULES.md#teams)) |
| Tasks | 6: 2 public (`bolt_to_tray`, `bolt_and_nut_to_fixture`) and 4 hidden (75% of the total) |
| Judged run | 4 repetitions per task, from unpublished starts that are the same for every team |
| Per repetition | 60 simulated seconds (1,200 steps at 20 Hz) and 90 s of cumulative `/act` time |
| Policy input | Instruction, current and goal JPEGs for all three 640×360 RGB cameras (`wrist`, `left`, `right`), proprioception |
| Policy output | Joint deltas plus a gripper target, one action or a chunk of up to 16 |
| Score | Finish, then precision, then speed: completed = 70 + precision × (5 + speed bonus), not completed under 15 ([scoring](RULES.md#scoring)) |
| Approach | One track, any approach: any model that works from the cameras can drive the arm. Training or fine-tuning one is encouraged ([approach](RULES.md#approach)) |
| Submissions | Accepted as soon as the repo and model checks pass; one per team per hour; your best score counts on one leaderboard |
| Real-arm final | Top 3 teams by simulation score at the cutoff run their best submission on the real arm ([final](RULES.md#real-arm-final)) |

Every number is in the [limits table](RULES.md#limits). Cutoff and results times are in the dashboard.
