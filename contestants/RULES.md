# Event rules and policy limits

The organizers' decisions on eligibility and conduct are final.

## Teams

| Rule | Value |
|---|---|
| Size | **Exactly 4 people, including the lead**, formed in EngineExchange. No 5th member, no smaller teams |
| Submitting | A team can submit once all 4 members are approved and checked in at the venue |
| Changes | Until submissions close you can create, join or leave a team; the lead can invite or remove members |
| Roster lock | Teams lock at the submission cutoff (Sunday 12:30) |
| Which team | A submission counts for the team you are on when you press submit; accepted runs stay with the team that submitted them |
| Shared | Scores, the hourly cooldown and the failure budget belong to the team |

## Tracks

Every submission has a track; an undeclared track counts as Open. Main-track submissions also count on the overall board.

| | Main track: **Model Training** | Side track: **Open** |
|---|---|---|
| Eligible | The arm is driven by a model your team trained or fine-tuned (VLA, VLM, learned visuomotor policy) that generalises across scenes. Hand-written control, even with learned perception, is Open | Any solution |
| Method note | **Required** (1–500 characters): what you trained and on which data | Optional (up to 500 characters) |
| Ranked by | Simulation score of your best Model Training submission | Judges, with the rubric below |
| Prize path | Top 3 go to the [real-arm final](#real-arm-final) | One winner, who gets an unranked demo on the real arm |

Organizers may move a submission to the other track; its score is kept. The overall board ranks every
submission whatever its track. Main-track finalists publish their training code, and organizers may
re-run a submission (for example with altered weights) to verify that its trained model drives the arm.

Open rubric (each criterion 1–5; ties go to Results, then Creativity):

| Criterion | Weight | What earns a 5 |
|---|---|---|
| Results | 35% | Top-tier sim score, robust across scenes and disturbances |
| Creativity | 30% | A genuinely new approach or a surprising, well-motivated use of the tools |
| Technical depth | 20% | Deep, well-measured work; ablations or analysis that teach something |
| Real-world readiness | 15% | Tolerant of calibration error, camera drift and part variation |

## Data, parts and models

- Allowed: any **public** data, data your team collects (sim or real), and pretrained models.
- At kickoff every team gets the same demonstration data and the **parts package**. Every task, public
  or hidden, uses only parts from the package, including parts no public task uses.
- Weights live in one public, ungated Hugging Face model, pinned at submission and mounted read-only
  at `/weights`. Never in GitHub. Details: [your model](SUBMIT.md#your-model).

## Code

If it's Python and runs in the image, it works. Anything else is on you.

You submit Python source; nothing is installed from your repository and prebuilt binaries are rejected.
Your entry point is a `Policy` class the image serves, or your own server implementing the
[policy HTTP contract](../docs/api.md). The policy container has no network. Details:
[SUBMIT.md](SUBMIT.md#you-submit-python-source-only-nothing-is-installed).

## Task and scenario shape

A judged run is 6 tasks × 4 repetitions = 24 repetitions: the two public tasks first, then the four
hidden ones. The 4 starts of a task differ from each other, are the same for every team and submission,
and are not your practice starts: part poses differ within the ranges the practice seeds cover. Some
judged repetitions vary the scene more and move a part during the episode, so act on what you see.

A repetition ends on success, after **60 simulated seconds** (1,200 steps at 20 Hz), or when its
**90 s cumulative `/act` budget** runs out. Running out of steps is a normal end, scored on the state
reached.

Hidden tasks reach your policy under anonymous ids `"1"`–`"4"`. You may branch on `task_id`, but do not
try to learn hidden task names or contents.

### The parts

ISO hex bolts M16×50, M24×70 and M30×100; hex nuts M24, M30 and M36; an M33 washer; smaller nuts and
washers (M12–M20) as clutter. Each part is nylon, stainless steel, brass or aluminium, chosen per seed,
so the same bolt weighs from about 15 g to 0.8 kg and looks different. The printed fixture has a tray,
three vertical holes (one per bolt, chamfered) and three hex pockets (one per nut). Dimensions are in
`engine/rc_engine/assets/`.

### Instructions

Every step carries one phrasing from the task's pool, chosen per episode, and a goal image of that
episode's scene finished. Phrasings that name a material, colour or position ("the brass bolt", "the
bolt nearer the tray") are used only when they pick out exactly one part. Hidden tasks phrase
instructions the same way and may tell parts apart by material alone. The public pools are listed
[below](#public-instruction-pools); `/tasks` serves them too.

### Scene variation

The seed sets, within the public ranges in `engine/rc_engine/assets/variation/public.json`:

- part poses (several cm, tens of degrees) and start states: a bolt may lie on its side, stand on its
  head or lean against the fixture; a nut may stand on a flat; in `bolt_to_tray`, whether the part is
  the bolt or the washer;
- each part's material (mass and look);
- lighting, table and fixture tints, the room (one of three);
- the two tripod cameras' poses (±1.5 cm, ±1.5°); the wrist camera is fixed;
- the fixture's position and turn (±4 cm, ±15°);
- up to two extra small parts. An extra part or decoy in a goal's seat scores the repetition 0.

The same seed and actions give identical pixels on every run. The `nudge` option (some judged
repetitions; in practice `"options": {"nudge": true}` on `/reset`) moves a part lying flat 3–6 cm and
up to 45°, 1.5–3.5 s into the episode, never while the gripper holds it.

A part is **placed** when it rests, released, in its seat: inside the tray; standing head-up in its own
hole; or flat in its own hex pocket with the flats aligned. A part in another part's seat scores nothing.

### Scoring

Finish the task, place every part precisely, then get faster. Points per repetition (scorer
`engine-v2`); a task has n = 1 or 2 placements. Energy and effort are not scored.

- **Completed** (every placement made and held, no decoy in a seat): 70 + precision × (5 + speed bonus).
  The speed bonus is +10 points every time you halve your time from the full 60 s, scaled by precision;
  there is no cap. In symbols: 70 + Q × (5 + B), B = 10 × ln(1200 / t) / ln 2, t = control steps used
  through the end of the success hold.
- **Precision** Q (0–100%): each required measurement (position, and orientation where it matters)
  gets full credit inside an ideal band, falling linearly to 0 at the pass tolerance. Q is the worst
  measurement over every part and the whole hold; public results name it ("limited by"). The bands are
  provisional and may be recalibrated on real finishes.
- **Not completed**: under 15 points for partial progress. Each placement made counts 15 / n; each one
  not made adds up to 4.29 / n for distance closed toward its seat beyond the first 1 cm. Ordered tasks
  count only placements made in order. Neither public task is ordered: `bolt_and_nut_to_fixture` accepts
  either order (`/tasks` describes each task).

| Time (steps) | 100% precision | 50% precision | 0% precision |
|---|---|---|---|
| 60 s (1,200) | 75.00 | 72.50 | 70.00 |
| 30 s (600) | 85.00 | 77.50 | 70.00 |
| 15 s (300) | 95.00 | 82.50 | 70.00 |
| 7.5 s (150) | 105.00 | 87.50 | 70.00 |

A task's score is the mean of its 4 repetitions. In the contest total each public task has weight 1 and
each hidden task has weight 1.5, so the hidden tasks carry 75%:
total = (p1 + p2 + 1.5 × (h1 + h2 + h3 + h4)) / 8.

## Leaderboard

- Your team's best score counts; a later, lower score never replaces it.
- Scores are rounded half up to two decimals (83.785 → 83.79) and ranked at that precision. Equal
  scores share a rank (1, 1, 3).
- After a run you see each public task's score, each hidden task's score under its anonymous id, the
  total, your rank, per-repetition results for public tasks, and your policy's
  output from loading, warmup and the public tasks. Hidden task names, hidden repetitions and hidden-task output are never
  shown ([what you see](SUBMIT.md#what-you-see-before-during-and-after-the-hidden-tasks)).

## Observation cameras

`three_view`: `images.wrist`, `images.left`, `images.right`, JPEGs decoding to 640×360 RGB (wrist bracket
plus two tripods). The goal image comes with every step. Proprioception: six joint positions and
velocities (shoulder pan, shoulder lift, elbow, wrist 1–3), gripper opening, palm position and
quaternion (wxyz). Wire details: [API reference](../docs/api.md).

## Action boundary

Return one `joint_delta` or an `action_chunk` of 1–16: six joint increments plus a gripper target
(0 closed, 1 open). The arm follows the UR12e's published speeds and default safety limits:

| Limit | Value | What happens |
|---|---|---|
| Joint step, shoulder pan and lift | ±0.10472 rad per step (120°/s) | Clamped, counted |
| Joint step, elbow and wrists | ±0.15708 rad per step (180°/s) | Clamped, counted |
| Joint target | ±2π; elbow ±160° | Clamped, counted |
| Gripper | [0, 1] | Clamped |
| TCP and elbow speed | 1.5 m/s | Whole step scaled down, direction kept, counted |
| TCP force, momentum, power | 150 N, 25 kg·m/s, 300 W | Measured only; a public notice says it "would trigger a safety stop on the real arm". Not scored |

- A malformed action (bad JSON, missing or wrong-length fields, non-finite values, wrong chunk length,
  echo mismatch) does not stop the run: that step is **held** (no joint motion, previous gripper target).
- Unknown top-level fields are ignored and listed once on your result page.
- A response over 2 MiB ends that repetition (during warmup it fails the run).
- Judged `/act` calls are never retried.

Your public results break clamps down by limit and joint ("wrist_1 limit 312 of 3,600 steps").

## Real-arm final

| Step | Detail |
|---|---|
| Who | Top 3 main-track teams by simulation rank at the cutoff, after eligibility checks |
| What runs | Your best main-track submission, unchanged |
| Runs | Once per public task, same `/act` contract; the goal image is a photo of the finished real scene |
| Ranking | Mean real-arm score; simulation score breaks ties |
| Open winner | One unranked demo run |

Real parts can differ in mass and look from the simulated ones.

## Limits

Container sizes (memory, CPU) are being finalised for the H100 and **may change before the event**. The
organizers publish the final values on the dashboard; until then, plan around the values below.

| Limit | Value |
|---|---|
| Repetition length | **60 simulated s** (1,200 steps at 20 Hz) |
| `/act` budget | **90 s cumulative** per repetition, transport included (runner and simulator time excluded); each call capped by what remains; connect ≤ 5 s; no retries |
| Startup plus warmup | 30 min per policy start (3 untimed warmup `/act` calls). The policy restarts before each task |
| Program-form server | Listening within 10 min of start |
| Total judged run | 90 min |
| `/act` response | 2 MiB; 1–16 actions |
| GPU | NVIDIA H100 (80 GB), amd64 |
| Memory | One combined CPU + GPU limit; over it ends the run (`policy_memory_exceeded`, limit and peak shown). The value is being finalised; it will be on the dashboard |
| CPUs, processes | 8 CPUs, 1024 processes |
| Writable space | `/scratch` 8 GB tmpfs (the only writable path); `/dev/shm` 1 GB |
| Repository | 200 MB total, 2000 files, 20 MB per file |
| Model snapshot | 100 GiB; 30 min download budget |
| Concurrent preparations | **1 per team**; **2 downloads at once** across all teams; a wait over **60 minutes** expires uncharged |
| `preparing` expiry | **30 minutes** from when a download slot is granted |
| Cooldown | 1 accepted submission per team per hour |
| Failure budget | 8 team-caused failures per rolling hour |
| Published policy log | First 50 and last 150 lines (≤ 32 KiB) per policy start; ≤ 8 starts and 128 KiB per submission |
| Practice runner | 10 s per `/act` call, one retry |

## Public instruction pools

`bolt_to_tray`, even seeds (the M30×100 bolt):

- Pick up the bolt and place it in the tray.
- Put the bolt in the tray.
- Drop the bolt into the tray.
- Move the bolt to the tray.
- Place the bolt in the tray.
- Pick up the bolt and put it in the tray.
- Move the gold bolt to the tray.
- Put the gold bolt in the tray.
- Put the {material:bolt} bolt in the tray.
- Pick up the {material:bolt} bolt and place it in the tray.

`bolt_to_tray`, odd seeds (the M33 washer):

- Pick up the washer and place it in the tray.
- Put the washer in the tray.
- Move the washer to the tray.
- Place the washer in the tray.
- Pick up the washer and put it in the tray.
- Put the white washer in the tray.
- Put the {material:washer} washer in the tray.
- Move the {material:washer} washer to the tray.

`bolt_and_nut_to_fixture`:

- Insert the smaller bolt into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Put the smaller bolt in the hole that matches its size and the nut in the hex pocket that matches its size.
- Seat the smaller bolt in its matching hole and the nut in its matching hex pocket.
- Insert the smaller bolt into its hole and seat the nut in its hex pocket.
- Place the nut in the hex pocket that matches its size and the smaller bolt in the hole that matches its size.
- Put the smaller of the two bolts in its matching hole, then the nut in its matching hex pocket.
- Insert the bolt nearer the tray into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Seat the gold nut in its hex pocket and the smaller bolt in its hole.
- Insert the {material:small_bolt} bolt into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Seat the {material:nut} nut in its hex pocket and the smaller bolt in its hole.

`{material:<part>}` becomes nylon, stainless steel, brass or aluminium. `small_bolt` is the M16×50.
