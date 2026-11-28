# Practice on the public tasks

The `practice` tool runs your repo and model in the **published policy image with the judged loader,
mounts, checks and offline settings**, against the public engine. Hardware, timing and memory are your
machine's, and hidden tasks are not run. Practice is optional.

## What you need

- **Docker**, about 16 GB of RAM and about 20 GB of free disk (the policy image is about 5.5 GB
  compressed and is pulled on the first run, plus the engine image).
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** and **git** (practice runs a commit).
- **An NVIDIA GPU, if you can**, for the engine and your policy. CPU rendering is about 3 steps/s
  (6 minutes per 1,200-step repetition); EGL on a GPU is about 30× faster. On macOS the engine also
  runs natively, without Docker, at about 20–30 steps/s; see [Without the image](#without-the-image).

On an arm64 machine (Apple silicon, DGX Spark) practice runs the `linux/arm64` build of the same release
(package set `runtime/locks/requirements.arm64.txt`, the same pinned versions); judging always runs the
`linux/amd64` build (`runtime/locks/requirements.amd64.txt`).

The image comes from the release manifest the bundle ships as `runtime/release.json`. When organizers
publish a newer one, replace the file or pass `--release FILE`.

## Run it

From the root of the unpacked participant bundle:

1. Create the practice environment (once):

   ```sh
   (cd runner && uv sync --group practice)
   ```

2. Build and start the public engine (drop the `--gpus`/`-e` flags for CPU rendering):

   ```sh
   docker build -t rc-engine:dev engine/
   docker run -d --rm --name rc-engine -p 127.0.0.1:8090:8090 \
     --gpus all -e MUJOCO_GL=egl -e NVIDIA_DRIVER_CAPABILITIES=all rc-engine:dev
   ```

3. Commit the bundle if it is not a git repository yet:

   ```sh
   git init -q && git add -A && git commit -q -m "Reality Check practice bundle"
   ```

4. Run the starter's example policy:

   ```sh
   python3 starter/scripts/make_example_weights.py /tmp/example-weights
   runner/.venv/bin/python -m practice run --repo starter --gpus none --weights-dir /tmp/example-weights --seeds 0-1
   ```

5. Run your own policy with the model you will submit:

   ```sh
   runner/.venv/bin/python -m practice run --repo /path/to/your/policy \
     --model https://huggingface.co/you/model [--revision <sha|branch|tag>] [--seeds 0-3] \
     [--gpus none]   # --gpus none on a Mac or any machine without an NVIDIA GPU
   ```

| Option | Effect |
|---|---|
| `--model URL` | Downloads the pinned snapshot from Hugging Face and mounts it read-only at `/weights` with the repo's folder layout, as every judged run does |
| `--weights-dir DIR` | Uses local weights instead |
| `--seeds 0-3` | Public seeds to play (four different starts) |
| `--tasks bolt_to_tray` | Plays only these public tasks (comma-separated) |
| `--judged` | Select judged episode settings in the practice launcher |
| `--scenario-seconds S` | With `--judged`, the `/act` budget per repetition (default 90); not the 60 s simulated length |
| `--max-steps N` | Ends each repetition after N steps, for a short CPU run (judged: up to 1,200) |
| `--gpus none` | Runs the policy without a GPU (default: all GPUs) |
| `--allow-dirty` | Practices the last commit despite uncommitted changes |
| `--print-argv` | Prints the exact `docker run` arguments |

## What it prints

- The provenance triple (Git SHA, Hugging Face SHA, image digest) and the release manifest used.
- The same repository and snapshot checks the backend runs; it refuses what the backend would reject.
- A table with one row per repetition (task, seed, result, steps, points, notes such as precision and
  the limiting measurement), with clamps by limit and joint, held steps and safety-limit notices
  indented under the row. Then each public task's mean.
- A resource summary: load time, peak memory (CPU + GPU, measured like the judged runner), `/act`
  p50/p95/max and `/act` seconds per repetition against the 90 s budget.
- A last line starting `VERDICT:`, for example
  `VERDICT: public mean 83.50 pts · 8 repetitions · 2 with clamps · 0 safety notices`, or why there is
  no score. Common setup problems (Docker not running, no disk space, image not pulled, engine not
  answering on :8090, a missing `--weights-dir`, uncommitted changes) print a `Next step:` line first.
- On stdout, one JSON object with the score (`total`, `breakdown`), `provenance` and per-repetition
  `episodes`.

Practice and judging allow 90 s of cumulative `/act` request time per schema v3
repetition. Transport time counts toward that budget. Each call gets one attempt.

## Train or export inside the image

The image runs as uid 10001 and offline. To download, train or export with it on your own machine,
turn the offline switches off and give it a writable Hugging Face cache:

```sh
docker run --rm --gpus all --user $(id -u):$(id -g) --shm-size=8g \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v $PWD/hf-cache:/scratch/hf -v $PWD/out:/out <image> python ...
```

`<image>` is `ghcr.io/lucasucal/rc-policy@` plus `image.manifest_digest` from `runtime/release.json`.
The image keeps its Hugging Face cache in `/scratch/hf`; the `-v $PWD/hf-cache:/scratch/hf` mount
keeps downloads between runs. `--shm-size` keeps DataLoader workers alive (Docker's default `/dev/shm` is 64 MB; or pass
`num_workers=0`). Leave these settings out of your policy: judging is offline.

## Without the image

On macOS the public engine runs natively, a fast way to iterate (same API on port 8090; use it instead
of step 2):

```sh
cd engine && uv sync && MUJOCO_GL=cgl .venv/bin/python -m rc_engine
```

`python3 scripts/selftest.py --weights DIR` in your starter copy checks the manifest, the HTTP contract,
and your `load()` and `act()` on one observation (see the [starter README](../../starter/README.md)). To drive
a policy server you started yourself (`--max-steps 240` for a short run):

```sh
cd runner && uv run python -m rc_runner.practice \
  --engine http://127.0.0.1:8090 --policy http://127.0.0.1:8080 \
  --tasks bolt_to_tray,bolt_and_nut_to_fixture --seeds 0-3 --judged
```

This skips the image, mounts and checks, so it does not match judging.

## The cell

The simulator is the real-arm cell, measured from LiDAR scans of the venue and a camera calibration
(`engine/rc_engine/assets/cell/cell.json`):

- **Robot and table.** The UR12e stands on its own pedestal beside a 2.5 × 1.06 m dark wood table. Its base
  plane is 0.187 m above the tabletop.
- **Fixture.** The cyan fixture's nominal pose is the real one's: square to the table, about 0.59 m in front
  of the base, tray to the robot's left. Episodes vary it ([scene variation](RULES.md#scene-variation));
  in the real-arm final it stays at the nominal pose.
- **Side cameras.** The left and right cameras are the real Gemini 336s at their calibrated poses (54.9°
  vertical FOV).
- **Wrist camera.** The wrist camera is the Gemini 335Lg on the bracket (61.1° vertical FOV).
- **Motion.** Every joint's step is clamped to 0.05 rad per 50 ms (1 rad/s), as on the real arm.
- **Proprioception.** `ee_pos` and `ee_quat` are in the robot base frame (UR `base_link`), as the real arm
  reports them.

Each frame is 640×360. No distortion, exposure, rolling shutter, depth or sensor noise is modelled.

## Rehearse on the real cell

The episode's room is drawn by the seed from four rooms. One of them, `real_cell`, is the venue of the
real-arm final: the team's scans of the room, with the backgrounds the two side cameras see textured from
their own frames. To play every repetition there, pass the room to the practice runner:

```sh
cd runner && uv run python -m rc_runner.practice --engine http://127.0.0.1:8090 \
  --policy http://127.0.0.1:8080 --room real_cell --seeds 0-3
```

or send `"options": {"room": "real_cell"}` in the engine's `/reset` body yourself. The option is for practice:
judging draws the room by the seed, and refuses it with the eval profile.

Some of the cell is still provisional. The robot's place in the scans is fitted from the scans until a
gripper touch-off on the real arm, and the side cameras' poses follow it, so they may move by a few
centimetres. Where the real arm stood during the capture, the side views show the scan instead of the
camera frame.
