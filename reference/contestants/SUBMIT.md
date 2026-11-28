# Submit a policy

You submit two things: a **public GitHub repository** with your Python code and a **public, ungated
Hugging Face model** with your weights. Judging runs them in the published policy image, offline, on an
NVIDIA RTX PRO 6000 (amd64). See the [limits table](RULES.md#limits).

### Steps

1. Copy [`starter/`](../../starter/README.md) into a new repository and set `entry_point` in `manifest.json`.
2. Run `python3 scripts/selftest.py --weights DIR` (checks the manifest and HTTP contract, then runs your `load()` and `act()` on DIR).
3. Upload your weights to your own public, ungated Hugging Face model repo ([your model](#your-model)).
4. Practice with the real model in the judged image ([PRACTICE.md](PRACTICE.md)). Optional, but it
   catches most failures before they cost a cooldown.
5. Push to public GitHub. In EngineExchange create or join a team ([teams](RULES.md#teams)), open
   **Policy runs** and submit: repo URL and
   branch, tag or commit SHA; model link and optional revision. There is no track to declare and no
   method note.
6. The submission is accepted as soon as the repo and model checks pass (seconds; the model is not
   downloaded at submit). The hourly cooldown starts then.

A public repo exposes every branch and commit pushed to it. To publish only the submitted commit, push
it to a fresh repo (`git push <fresh-remote> HEAD:main`).

### Repository layout

```text
your-policy-repo/        # public GitHub repo, Python only
  manifest.json          # {"contract": "v0", "schema_version": 3, "entry_point": "policy.example:ExamplePolicy"}
  policy/
    example.py
```

`manifest.json` is strict JSON. Set `contract` to `"v0"`. A class policy must declare
`"schema_version": 3`. A program reports the schema in `/readyz`. Choose one of the
following forms for `entry_point`.

| Form | `entry_point` | Who serves HTTP |
|---|---|---|
| Class (starter default) | `"package.module:Attr"`: a `rc_sdk.policy.Policy` subclass, or a zero-argument function returning one | The image serves `/healthz`, `/readyz`, `/act` around it |
| Program | A dotted module (`"mypkg.server"`) or a relative `.py` path (`"serve.py"`) | Your program, run as `__main__` |

A program must listen on `0.0.0.0:$RC_POLICY_PORT` (8080) **within 10 minutes**, answer `/healthz` at
once, report `ready: true` on `/readyz` only after loading, and keep running. Start the server first,
then load. Examples: `starter/examples/server_fastapi/serve.py`, `starter/examples/server_stdlib/serve.py`
(a stdlib keep-alive server needs `TCP_NODELAY`, or each `/act` waits about 40 ms).

### Import paths, working directory and the sandbox

| Path | What it is |
|---|---|
| `/submission` | Your repo, read-only, working directory, appended **last** to `sys.path` (a `src/` layout works) |
| `/weights` | Your model snapshot, read-only (`RC_WEIGHTS_DIR`) |
| `/scratch` | The only writable path (`TMPDIR=/scratch/tmp`), empty at every start |

- No network. Load everything from `/weights` by path.
- `rc_sdk` comes from the image; your copy is replaced. An entry package named like an image package
  (`torch`, `lerobot`, `rc_sdk`) is rejected; other colliding folders are ignored with a warning, so
  rename vendored copies.
- `torch.load` defaults to `weights_only=True`; prefer safetensors.
- Print errors to stdout/stderr: you see that output for loading, warmup and public tasks.

### You submit Python source only; nothing is installed

If it's Python and runs in the image, it works. Anything else is on you.

- Nothing is installed from `requirements.txt`, `pyproject.toml` or `setup.py`; a Dockerfile is ignored.
  The package set is the image's lock file (`runtime/locks/requirements.amd64.txt`). Pure-Python code
  can be vendored into your repo. To request a package, ask the organizers.
- Rejected before any cooldown is charged: native binaries (`.so`, `.pyd`, `.dylib`, `.dll`, `.exe`,
  `.o`, `.a`, `.whl`, ELF/Mach-O/PE content), compiled GPU code (`.ptx`, `.cubin`, `.fatbin`), archives
  (`.zip`, `.tar`, ...) and symlinks.
- Code generated at run time by image packages (`torch.compile`, Triton, JAX, numba) is fine.
- The submit and practice checks warn about imports that are neither in the image nor in your repo
  ([KSI-019](KNOWN-ISSUES.md)), hub ids loaded by name, and servers bound to localhost.

### flash-attention to PyTorch SDPA

flash-attn is not in the image ([KSI-003](KNOWN-ISSUES.md)). SDPA computes the same attention and
checkpoints load unchanged:

- `attn_implementation="flash_attention_2"` → `"sdpa"` (also in `config.json`).
- `flash_attn_func(q, k, v, causal=c)` →
  `F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=c).transpose(1, 2)`.
- Variable-length batches → pad and pass `attn_mask`.

### Your model

Gated model repositories cannot be submitted, even if you have access to them. Upload the weights
your policy needs to your own public, ungated Hugging Face model repository, including when you use a
pretrained model unchanged.

At submit the model is checked by metadata only: public, ungated, a model repo, revision pinned to a
commit, file names and sizes. Nothing is downloaded and the submission is accepted at once. Every judged
run then downloads the pinned commit fresh from Hugging Face, checks each file against the list recorded
at submit, runs the content checks, mounts the files read-only at `/weights` and deletes them after the
run. Nothing is cached between runs.

| Rule | Detail |
|---|---|
| One repo | One **public, ungated** Hugging Face model repo holds **everything your policy loads**: weights, config, tokenizer and processor files, and any other model it uses. Organizers provide no base models |
| Formats | safetensors recommended; `.pt`/`.pth`/`.bin`/`.ckpt` work (`torch.load` defaults to `weights_only=True`). Rejected by file name at submit: archives (`.zip`, `.tar`, `.gz`, `.7z`, ...), TensorRT engines (`.engine`, `.plan`, `.trt`), compiled GPU code (`.ptx`, `.cubin`, `.fatbin`) and native binaries (`.so`, `.dll`, ...). Upload the files themselves, not an archive of them |
| Layout | `/weights` is the repo root, folders kept: repo `config.json` → `/weights/config.json`, repo `policy/model.safetensors` → `/weights/policy/model.safetensors` |
| Read-only, by path | `/weights` (`RC_WEIGHTS_DIR`) is read-only; write only to `/scratch`. Load with `from_pretrained("/weights/...")` or an explicit file path |
| Offline | `HF_HUB_OFFLINE=1`. A hub id loaded by name fails (`OSError`). Disable pretrained downloads in configs (for example `pretrained_backbone_weights=None`) |
| Several models | A base model plus an adapter, a detector plus a VLA, someone else's pretrained model: download them and upload them into your one repo, each in its own subfolder ([example](#several-models-in-one-repo)) |
| LoRA | Upload the **merged** model (QLoRA: merge onto a BF16 base), or the base and the adapter as two subfolders. An adapter pointing at another repo does not resolve |
| Licences | Re-upload only models whose licence permits redistribution in a public, ungated repo, including unchanged pretrained models, and keep the original licence and attribution files alongside the weights. Models distributed gated (Llama, Gemma, PaliGemma) often restrict redistribution: check the licence first. If it does not allow public redistribution, that model cannot be used |
| Not in GitHub | Weights never go in the code repo (20 MB per-file cap) |
| Revision | Optional: a full 40-character commit SHA, a tag or a branch. A tag or branch is resolved at submit and pinned to that commit; later pushes do not change the submission. Omitted = current head |
| Keep it available | Every judged run downloads the pinned commit again. Until results are final keep the repo public and ungated, and do not delete it or rewrite the pinned commit (force-push, squash, history reset). Otherwise the run fails with `weights_unavailable`, charged |
| Size | 100 GiB per snapshot. The download is part of each judged run: it counts toward the 90-min run deadline, not the 30-min startup limit, and a larger snapshot takes longer. Over the cap, or to save time: publish an inference branch with only the files your policy loads and submit it as the revision |

Upload (a write token from huggingface.co/settings/tokens; the repo is created public):

```sh
pip install -U huggingface_hub
hf auth login
hf upload <org>/<model> OUT_DIR .
```

Test exactly what will be mounted, at the revision you will submit:

```sh
hf download <org>/<model> --revision <rev> --local-dir /tmp/snapshot
python3 scripts/selftest.py --weights /tmp/snapshot     # in your policy repo; must end with ALL PASS
runner/.venv/bin/python -m practice run --repo /path/to/policy \
  --model https://huggingface.co/<org>/<model> --revision <rev>   # from the bundle root
```

Practice downloads the snapshot from Hugging Face and mounts it at `/weights` the way a judged run does
([PRACTICE.md](PRACTICE.md)).

#### Several models in one repo

For example, a policy that loads `lerobot/smolvla_base` and your own checkpoint:

```sh
hf download lerobot/smolvla_base --local-dir OUT/smolvla_base   # Apache-2.0: redistributable
cp -r path/to/your/checkpoint OUT/policy
hf upload <org>/<model> OUT .
```

```python
import os
from pathlib import Path

WEIGHTS = Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")

def load(self):
    base = WEIGHTS / "smolvla_base"    # was hub id "lerobot/smolvla_base"
    policy = WEIGHTS / "policy"
    ...
```

Configs inside a model can name further hub ids (SmolVLA's `vlm_model_name` is
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`; PEFT's `adapter_config.json` has
`base_model_name_or_path`). Download those too and point the field at the `/weights/...` path. The
submit checks warn about hub ids your code loads by name.

**Load time counts six times.** The policy restarts before each task (6 starts per judged run, all
inside the 90-min run), so keep load plus warmup well under the 30-min startup limit. The model is
downloaded once per run, before the first start, and reused by every restart.

To export inside the image on your own machine, see [PRACTICE.md](PRACTICE.md#train-or-export-inside-the-image).

**When the model fails.** A missing, private, gated, empty or oversized model, an unknown revision or a
rejected file name is refused at submit: no cooldown, but it counts as a team-caused failure. Problems
found when a run downloads and checks the model are in the [failure table](#what-you-see-before-during-and-after-the-hidden-tasks).

### Provenance on every submission

The submission page shows the Git commit SHA, the Hugging Face snapshot SHA and the policy image digest
that ran. Practice prints the same three values.

## Cooldown, cutoff and status

- One accepted submission per team per hour, starting when repo and model checks pass. Failed checks do
  not start it; 8 team-caused failures per rolling hour are allowed.
- A failed run spends the cooldown unless its code is uncharged (table below). The failure says whether
  it was charged.
- The cutoff is judged by when you press submit; accepted work may finish after it.
- Status: received → validating → queued → starting → running → scored or failed.

### What you see before, during and after the hidden tasks

| Phase | Shown |
|---|---|
| Running, public tasks | Phase, deadlines, `/readyz` state; one row per public repetition: success, steps, precision, points, limiting measurement |
| Running, hidden tasks | Only the phase and when it started |
| Scored | Total, each public task's score, each hidden task's score under its anonymous id; public rows; clamp and safety notices; load time, peak memory, `/act` p50/p95; summed hidden counts (early ends, held steps) |
| Failed | Stage, code, explanation, whether charged, next steps, a [known issue](KNOWN-ISSUES.md) link when one matches. During the hidden tasks: only code and stage |
| Policy log | Your stdout/stderr from loading, warmup and public tasks (print with `flush=True`). Hidden-task output is never shown |

A crash, timeout or error inside a judged repetition ends only that repetition; the policy restarts and
the run continues. Going over the memory limit fails the run.

| Code | Charged | Meaning |
|---|---|---|
| `container_unhealthy` | yes | The container exited or never listened |
| `weights_load_failed` | yes | Import, entry point or `load()` failed |
| `warmup_timeout` | yes | Not ready within the load deadline, or a warmup `/act` timed out |
| `policy_memory_exceeded` | yes | Over the combined CPU + GPU memory limit |
| `policy_protocol_violation` | yes | Wrong contract on `/readyz`, or a warmup response over 2 MiB |
| `motion_limit_violation` | yes | Legacy; malformed actions are now held steps |
| `run_deadline_exceeded` | yes | The whole run went over 90 minutes |
| `weights_unavailable` | yes | At run time the model repo was deleted, made private or gated, or the pinned commit no longer exists |
| `weights_unsafe_entries`, `weights_check_failed` | yes | The downloaded model failed the content checks (unsafe entries, rejected formats) |
| `weights_snapshot_too_large` | yes | The downloaded model is over 100 GiB (Hugging Face did not report every file size at submit) |
| `infra_error` | no | Evaluation service error |
| `stopped` | no | An organizer stopped the run |
| `image_missing`, `image_mismatch`, `context_verify_failed` | no | Image or code could not be verified |
| `weights_verify_failed` | no | The downloaded files do not match the list recorded at submit |
| `weights_provider_unavailable` | no | Hugging Face outage, rate limit or timeout during the run's download; the run is retried |
| `evidence_recording_failed` | no | The result could not be recorded |

`weights_download_failed`, `weights_download_timeout` and the `preparation_*` codes appear only on older
submissions.

## Wire contract

Each schema v3 step carries an instruction, current and finished-scene `wrist`, `left` and
`right` RGB JPEGs at 640×360, and proprioception. Return six joint deltas within ±0.05 rad per
step and an absolute gripper target in `[0, 1]`, or a chunk of up to 16 such actions. See the
[action boundary](RULES.md#action-boundary) and [API reference](../docs/api.md).
