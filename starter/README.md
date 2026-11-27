# Reality Check policy starter

Copy this folder into a new public GitHub repository; it becomes your policy
project. Rules and limits are in [`contestants/RULES.md`](../contestants/RULES.md),
the full submission guide in [`contestants/SUBMIT.md`](../contestants/SUBMIT.md)
and the HTTP contract in [`docs/api.md`](../docs/api.md).

## The path

1. **Edit `policy/example.py`.** `load()` reads your model from `/weights`
   (your Hugging Face snapshot, mounted read-only); `act(obs)` returns one
   action per control step. `manifest.json` names the class:
   `"entry_point": "policy.example:ExamplePolicy"`.
2. **Upload your weights** to your own public, ungated Hugging Face model repo (safetensors
   recommended; no archives). Gated repos cannot be submitted. Upload pretrained weights even when unchanged.
   Everything `load()` reads goes in that one repo; the
   repo root is `/weights` and several models go in subfolders (`/weights/<subfolder>`,
   [your model](../contestants/SUBMIT.md#your-model)). Every judged run downloads the
   pinned commit again, so keep the repo public and the commit intact until results
   are final. To try the
   flow before you have a model, `python3 scripts/make_example_weights.py OUT_DIR`
   writes a checkpoint the example loads (standard library only; no torch needed);
   practice runs it with `--weights-dir OUT_DIR` and no `--model`.
3. **Test locally.** `python3 scripts/selftest.py --weights DIR` (your model snapshot) must end with
   `ALL PASS`: it runs your `load()` and `act()` on a real step-0 observation. `CONTRACT ONLY: …`
   means they did not run here, and says why and what to do; `FAILED: …` names the first failed check.
   Optional: run your real model on the public tasks with the practice tool
   ([PRACTICE.md](../contestants/PRACTICE.md)).
4. **Push and submit.** Push to public GitHub, then submit the repository URL,
   a branch, tag or commit, and the model link in the dashboard
   ([SUBMIT.md, Steps](../contestants/SUBMIT.md#steps)).

Never commit weights; `.gitignore` already excludes common checkpoint files.

## What is in here

| Path | What it is |
|---|---|
| `policy/example.py` | The policy you edit: a tiny linear model loaded from `/weights/model.safetensors` |
| `manifest.json` | `contract` and `entry_point`; the model link you submit in the dashboard selects the model |
| `scripts/selftest.py` | Checks the manifest, your entry point and the HTTP contract; never downloads a model |
| `scripts/make_example_weights.py` | Writes the example checkpoint |
| `scripts/sample_obs/` | The step-0 camera and goal JPEGs the self-test sends to your `act()` |
| `rc_sdk/` | Serves `/healthz`, `/readyz` and `/act` around your `Policy` class. Keep it; judging uses the image's own copy |
| `examples/` | Optional alternatives, below |

## Optional

- **Serve HTTP yourself.** Set `entry_point` to a program (`"serve.py"`) that
  listens on `0.0.0.0:$RC_POLICY_PORT` and implements the contract. Copy both
  files of `examples/server_fastapi/` or `examples/server_stdlib/` to the
  repository root. The self-test starts both examples and checks them.
- **LeRobot SmolVLA.** `examples/smolvla/smolvla.py` wraps `lerobot/smolvla_base`.
  Copy it to `policy/` and set `entry_point` to
  `policy.smolvla:ExamplePolicy`.

The examples show the contract; they are not score baselines.
