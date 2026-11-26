# Submit a policy

You submit two things: a **public GitHub repository** with your Python code and a **public, ungated
Hugging Face model** with your weights. Judging runs them in the published policy image, offline, on an
NVIDIA H100 (amd64). Limits: [limits table](RULES.md#limits).

### Steps

1. Copy [`starter/`](../starter/README.md) into a new repository and set `entry_point` in `manifest.json`.
2. Run `python3 scripts/selftest.py --weights DIR` (checks the manifest and HTTP contract, then runs your `load()` and `act()` on DIR).
3. Upload your weights to one Hugging Face model repo ([your model](#your-model)).
4. Practice with the real model in the judged image ([PRACTICE.md](PRACTICE.md)). Optional, but it
   catches most failures before they cost a cooldown.
5. Push to public GitHub. In EngineExchange create or join a team (exactly 4 people including the lead;
   all 4 approved and checked in before it can submit), open **Policy runs** and submit: repo URL and
   branch, tag or commit SHA; model link and optional revision. There is no track to declare and no
   method note.
6. If the dashboard shows `preparing`, wait: the cooldown starts only when the model is downloaded and
   verified.

A public repo exposes every branch and commit pushed to it. To publish only the submitted commit, push
it to a fresh repo (`git push <fresh-remote> HEAD:main`).

### Repository layout

```text
your-policy-repo/        # public GitHub repo, Python only
  manifest.json          # {"contract": "v0", "entry_point": "policy.example:ExamplePolicy"}
  policy/
    example.py
```

`manifest.json` is strict JSON with two fields: `contract` (`"v0"`) and `entry_point`, in one of two forms:

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

flash-attn is not in the image ([KSI-027](KNOWN-ISSUES.md)). SDPA computes the same attention and
checkpoints load unchanged:

- `attn_implementation="flash_attention_2"` → `"sdpa"` (also in `config.json`).
- `flash_attn_func(q, k, v, causal=c)` →
  `F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=c).transpose(1, 2)`.
- Variable-length batches → pad and pass `attn_mask`.

### Your model

| Rule | Detail |
|---|---|
| One repo | One **public, ungated** Hugging Face model repo holds **everything your policy loads**: weights, config, tokenizer and processor files |
| Pinned | Resolved to a commit at submit time and pinned. Optional revision: a full 40-character SHA, a branch or a tag. Omitted = current head |
| Mounted | Read-only at `/weights` (`RC_WEIGHTS_DIR`). Load by path |
| Offline | `HF_HUB_OFFLINE=1`. A hub id loaded by name fails (`OSError`). Disable pretrained downloads in configs (for example `pretrained_backbone_weights=None`) |
| LoRA | Upload the **merged** model (QLoRA: merge onto a BF16 base), or base plus adapter in the same repo. An adapter pointing at another repo does not resolve |
| Several models | Subfolders of the one repo: `/weights/detector/`, `/weights/vla/` |
| Not in GitHub | Weights never go in the code repo (20 MB per-file cap) |
| Size | 100 GiB snapshot, 30 min download. Over it: publish a clean inference branch and submit it as the revision |
| Formats | safetensors recommended; `.pt`/`.bin`/`.ckpt` work. Archives, TensorRT engines and compiled GPU code are rejected |
| Base models | Only when organizers publish a curated base cache (`/hf-base`, listed in `runtime/release.json`). **None is published yet**: until then upload everything |

Upload (a write token from huggingface.co/settings/tokens; the repo is created public):

```sh
pip install -U huggingface_hub
hf auth login
hf upload <org>/<model> OUT_DIR .
```

**Load time counts six times.** The policy restarts before each task (6 starts per judged run, all
inside the 90-min run), so keep load plus warmup well under the 30-min startup limit.

To export inside the image on your own machine, see [PRACTICE.md](PRACTICE.md#train-or-export-inside-the-image).

**Preparing.** A model not yet cached is downloaded and verified before the cooldown starts; one
preparation per team at a time. Hugging Face outages cost nothing; a missing, private, gated, empty,
oversized or unsafe model counts as a team-caused failure.

### Provenance on every submission

The submission page shows the Git commit SHA, the Hugging Face snapshot SHA and the policy image digest
that ran. Practice prints the same three values.

## Cooldown, cutoff and status

- One accepted submission per team per hour, starting when repo and model checks pass. Failed checks do
  not start it; 8 team-caused failures per rolling hour are allowed.
- A failed run spends the cooldown unless its code is uncharged (table below). The failure says whether
  it was charged.
- The cutoff is judged by when you press submit; accepted work may finish after it.
- Status: `preparing` → received → validating → queued → starting → running → scored or failed.

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
| `weights_download_failed`, `weights_download_timeout` | yes | The pinned weights could not be fetched |
| `infra_error` | no | Evaluation service error |
| `stopped` | no | An organizer stopped the run |
| `image_missing`, `image_mismatch`, `context_verify_failed`, `weights_verify_failed`, `base_cache_mismatch` | no | Image, code or weights could not be verified |
| `weights_provider_unavailable` | no | Hugging Face outage |
| `evidence_recording_failed` | no | The result could not be recorded |

## Wire contract

Each step: instruction, goal image, `wrist`/`left`/`right` 640×360 RGB JPEGs and joint state in; six
joint deltas plus a gripper target in `[0, 1]` out (or a chunk of up to 16). Limits:
[action boundary](RULES.md#action-boundary). Full contract: [API reference](../docs/api.md).
