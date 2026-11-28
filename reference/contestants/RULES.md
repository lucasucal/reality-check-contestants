# Event rules and policy limits

The organizers' decisions on eligibility and conduct are final.

## Teams

| Rule | Value |
|---|---|
| Size | **3 or 4 people, including the lead**, formed in EngineExchange |
| Submitting | A team can submit once every member is approved and checked in at the venue |
| Changes | Until submissions close you can create, join or leave a team; the lead can invite or remove members |
| Roster lock | Teams lock at the submission cutoff (Sunday 12:30, London time) |
| Which team | A submission counts for the team you are on when you press submit; accepted runs stay with the team that submitted them |
| Shared | Scores, the hourly cooldown and the failure budget belong to the team |

## Approach

One track, any approach. Any model can drive the arm: a VLA, a world or action model, a visuomotor
policy, anything that works from the cameras. We encourage training or fine-tuning a model; Black Forest
Labs' FLUX models and LeRobot models (such as SmolVLA) are good places to start. There is no track to
declare and no method note. One leaderboard ranks every team by its best scored submission; the top 3
go to the [real-arm final](#real-arm-final).

## Data, parts and models

- Allowed: any **public** data, data your team collects (sim or real), and pretrained models.
- At kickoff every team gets the same demonstration data and the **parts package**. Every task, public
  or hidden, uses only parts from the package, including parts no public task uses.
- Upload all weights to your own public, ungated Hugging Face model repository, even if you use a
  pretrained model unchanged. Gated repositories cannot be submitted. The model is pinned at submission
  and mounted read-only at `/weights`. Never put weights in GitHub. Re-upload only models whose licence
  permits public redistribution, with their licence files. Details: [your model](SUBMIT.md#your-model).

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
judged repetitions vary the scene more than the public ranges, so act on what you see.

A repetition ends on success, after **60 simulated seconds** (1,200 steps at 20 Hz), or when its
**90 s cumulative `/act` budget** runs out. Running out of steps is a normal end, scored on the state
reached. A fourth end, the **protective stop**, exists only in a simulator started with its opt-in
enforcement (`RC_SAFETY_STOP=enforce`); it is **off in judged runs** unless the organizers announce
otherwise. When on, a 50 ms step over a UR12e safety limit (see "Action boundary") ends the repetition,
unsuccessful, scored on the state reached (`end_reason: "protective_stop"`).

Hidden tasks reach your policy under anonymous ids `"1"`–`"4"`. You may branch on `task_id`, but do not
try to learn hidden task names or contents.

### The parts

ISO hex bolts M16×50, M24×70 and M30×100; hex nuts M24, M30 and M36; an M33 washer; smaller nuts and
washers (M12–M20) as clutter. Every part is grey PLA, like the 3D-printed kit at the event, with about
a real print's mass (20% infill, 3 walls: an estimated 680 kg/m³, so the M30×100 bolt weighs about 65 g
and the M16×50 about 9 g). Parts differ only in shape and size. The printed fixture has a tray,
three vertical holes (one per bolt, chamfered) and three hex pockets (one per nut). Dimensions are in
`engine/rc_engine/assets/`.

**Parts and assets.** The fasteners and three of the four rooms come from
[SimReady Hardware](https://huggingface.co/datasets/projectsim/simready-hardware) by Project Sim
(Kaedim), revision `827f45cdffe63ae699351fdba58c50de73e4d40a`, CC-BY-4.0. The simulator's converted
copies are in `engine/rc_engine/assets/fasteners/` and `engine/rc_engine/assets/rooms/`; each folder's
`VENDORED.md` lists the source packages and dimensions. The fixture is original
(`engine/rc_engine/assets/fixture/`).

### Instructions

Every step carries one phrasing from the task's pool, chosen per episode, and three goal views of
that episode's finished scene. Phrasings name parts by kind, size or position ("the smaller bolt", "the
bolt nearer the tray"), never by colour or material; a position phrasing is used only when it picks out
exactly one part. Hidden tasks phrase instructions the same way and may tell parts apart by size alone.
The public pools are listed [below](#public-instruction-pools); `/tasks` serves them too.

### Scene variation

The seed sets, within the public ranges in `engine/rc_engine/assets/variation/public.json`:

- part poses (several cm, tens of degrees) and start states: a bolt may lie on its side, stand on its
  head or lean against the fixture; a nut may stand on a flat; in `bolt_to_tray`, whether the part is
  the bolt or the washer;
- one shade of grey shared by every part (a few per cent);
- lighting, table and fixture tints, the room (one of four; one is the venue of the real-arm final);
- the two tripod cameras' poses (±1.5 cm, ±1.5°); the wrist camera is fixed;
- the fixture's position and turn (±4 cm, ±15°);
- up to two extra small parts. An extra part or decoy in a goal's seat scores the repetition 0.

The same seed and actions give identical pixels on every run.

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

- One leaderboard. Your team's best score counts; a later, lower score never replaces it.
- Scores are rounded half up to two decimals (83.785 → 83.79) and ranked at that precision. Equal
  scores share a rank (1, 1, 3).
- After a run you see each public task's score, each hidden task's score under its anonymous id, the
  total, your rank, per-repetition results for public tasks, and your policy's
  output from loading, warmup and the public tasks. Hidden task names, hidden repetitions and hidden-task output are never
  shown ([what you see](SUBMIT.md#what-you-see-before-during-and-after-the-hidden-tasks)).

## Observation cameras

`three_view`: `images.wrist`, `images.left`, `images.right`, 640×360 RGB JPEGs. The same roles
appear in `goal` on every step as views of the finished scene. Proprioception: six joint positions
and velocities, gripper opening, palm position and quaternion (wxyz) in the robot base frame.
The scene is the real-arm cell ([the cell](PRACTICE.md#the-cell)).
Wire details: [API reference](../docs/api.md#request).

## Action boundary

Return one `joint_delta` or an `action_chunk` of 1–16: six joint increments plus a gripper target
(0 closed, 1 open). Simulation applies these limits.

| Limit | Value | What happens |
|---|---|---|
| Joint step, all six joints | ±0.05 rad per 50 ms step | Clamped, counted |
| Joint target | ±2π; elbow ±160° | Clamped, counted |
| Gripper | [0, 1] | Clamped |
| TCP and elbow speed | 1.5 m/s | Whole step scaled down, direction kept, counted |
| TCP force, momentum, power | 150 N, 25 kg·m/s, 300 W, each averaged over the 50 ms step | Measured only; a public notice says it "would trigger a safety stop on the real arm". Not scored. With the opt-in protective stop (off in judged runs) the first step over a limit ends the repetition instead |

- A malformed action (bad JSON, missing or wrong-length fields, non-finite values, wrong chunk length,
  echo mismatch) applies one **held** step with zero joint increments and the previous gripper target.
  An invalid chunk applies none of its elements.
- Extra fields at response or chunk level are ignored and listed once on your result page. Unknown
  fields in a `joint_delta` make the response malformed.
- A response over 2 MiB ends that repetition (during warmup it fails the run).
- Judged `/act` calls are never retried.

Your public results break clamps down by limit and joint ("wrist_1 limit 312 of 3,600 steps").

## Real-arm final

| Step | Detail |
|---|---|
| Who | Top 3 teams by simulation score at the cutoff, after eligibility checks |
| What runs | Your best submission, unchanged. Organizers may re-run it to verify it |
| Runs | Once per public task, same `/act` contract with photos of the finished real scene from all three camera views |
| Ranking | Mean real-arm score; simulation score breaks ties |
| Scene | The fixture is fixed at its nominal pose, square to the table where the simulator's default places it, with no variation |

The real parts are 3D-printed grey PLA copies of the package parts. Their exact mass depends on the
print and can differ from the simulated estimate.

## Limits

Container sizes (memory, CPU) are being finalised for the RTX PRO 6000 and **may change before the event**. The
organizers publish the final values on the dashboard; until then, plan around the values below.

| Limit | Value |
|---|---|
| Repetition length | **60 simulated s** (1,200 steps at 20 Hz) |
| `/act` budget | **90 s cumulative** per repetition, transport included (runner and simulator time excluded); each call capped by what remains; connect ≤ 5 s; no retries |
| Startup plus warmup | 30 min per policy start (3 untimed warmup `/act` calls). The policy restarts before each task |
| Program-form server | Listening within 10 min of start |
| Total judged run | 90 min, model download included |
| `/act` response | 2 MiB; 1–16 actions |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), amd64 |
| Memory | One combined CPU + GPU limit; over it ends the run (`policy_memory_exceeded`, limit and peak shown). The value is being finalised; it will be on the dashboard |
| CPUs, processes | 8 CPUs, 1024 processes |
| Writable space | `/scratch` 8 GB tmpfs (the only writable path); `/dev/shm` 1 GB |
| Repository | 200 MB total, 2000 files, 20 MB per file |
| Model snapshot | **100 GiB**. Downloaded fresh at the start of every judged run; the download counts toward the 90-min run, not the 30-min startup |
| Cooldown | 1 accepted submission per team per hour |
| Failure budget | 8 team-caused failures per rolling hour |
| Published policy log | First 50 and last 150 lines (≤ 32 KiB) per policy start; ≤ 8 starts and 128 KiB per submission |
| Practice runner | 90 s cumulative `/act` budget, one attempt per call |

## Public instruction pools

`bolt_to_tray`, even seeds (the M30×100 bolt):

- Pick up the bolt and place it in the tray.
- Put the bolt in the tray.
- Drop the bolt into the tray.
- Move the bolt to the tray.
- Place the bolt in the tray.
- Pick up the bolt and put it in the tray.
- Set the bolt down in the tray.
- Carry the bolt over to the tray.

`bolt_to_tray`, odd seeds (the M33 washer):

- Pick up the washer and place it in the tray.
- Put the washer in the tray.
- Move the washer to the tray.
- Place the washer in the tray.
- Pick up the washer and put it in the tray.
- Drop the washer into the tray.
- Set the washer down in the tray.
- Carry the washer over to the tray.

`bolt_and_nut_to_fixture`:

- Insert the smaller bolt into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Put the smaller bolt in the hole that matches its size and the nut in the hex pocket that matches its size.
- Seat the smaller bolt in its matching hole and the nut in its matching hex pocket.
- Insert the smaller bolt into its hole and seat the nut in its hex pocket.
- Place the nut in the hex pocket that matches its size and the smaller bolt in the hole that matches its size.
- Put the smaller of the two bolts in its matching hole, then the nut in its matching hex pocket.
- Insert the bolt nearer the tray into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Insert the shorter bolt into the hole that matches its size, then seat the nut in the hex pocket that matches its size.
- Seat the nut in its hex pocket and the smaller bolt in its hole.

The smaller (shorter) bolt is the M16×50.
