# Known issues

Last updated: 2026-11-25

Real issues in the judged environment (the published policy image on NVIDIA H100, amd64), with workarounds. Treat each entry as a lead and test your own policy with the practice tool.
IDs are stable and never reused. When a failed run matches an entry, your submission page links to it (not for failures during the hidden tasks).

## KSI-003 - FlashAttention 2 and 3 are not in the image

- Symptom: `import flash_attn` fails.
- Cause: the image ships no flash-attn; nothing is installed from your repository.
- Workaround: use PyTorch SDPA (`attn_implementation="sdpa"`); see "flash-attention to PyTorch SDPA" in [SUBMIT.md](SUBMIT.md) and KSI-027.
- Status: confirmed in the pinned image

## KSI-004 - torch.compile and Triton: supported, with cold-start compile time

- Symptom: the first call of a `torch.compile`d module or `@triton.jit` kernel takes seconds to minutes.
- Cause: compilation happens at first use, and the compile cache lives in `/scratch`, which is empty at every policy start (the policy restarts before each task). Real models can take over a minute (SmolVLA `sample_actions` took about 100 s).
- Workaround: compile and run one dummy call in `load()` so `/act` never pays for it; keep it inside the startup deadline. Do not set `TRITON_PTXAS_PATH` or `CC`.
- Status: confirmed in the pinned image

## KSI-006 - TensorFlow is not in the image

- Symptom: `ModuleNotFoundError: No module named 'tensorflow'`, or a model whose code needs TensorFlow (for example Octo) fails to import.
- Cause: TensorFlow (CPU and GPU) is not installed.
- Workaround: convert to PyTorch or JAX. Keras 3 runs on JAX (the image sets `KERAS_BACKEND=jax`) or torch. Octo's orbax parameters restore with `orbax.checkpoint`; vendor a pure Flax copy of the model code.
- Status: confirmed in the pinned image

## KSI-007 - bitsandbytes and 4-bit or 8-bit quantization

- Symptom: quantized layers are slower than expected or the actions drift.
- Cause: bitsandbytes is installed and its 8-bit and 4-bit layers run on CUDA, but accuracy and speed depend on your model. A practice team found int8 and nf4 saved memory but not time on MolmoAct2, and moved the actions noticeably. GPTQ and AWQ are not installed.
- Workaround: prefer bf16 or fp16 weights; if you quantize, measure your own model in practice.
- Status: confirmed in the pinned image (operations run); speed and accuracy reported by a practice team

## KSI-008 - Policy out of memory (`policy_memory_exceeded`)

- Symptom: the run fails with `policy_memory_exceeded`, `CUDA out of memory`, or exit code 137.
- Cause: the policy container has one combined CPU + GPU memory limit ([limits table](RULES.md#limits)); the runner adds resident and GPU memory about once a second and stops the container when the sum passes it. Your failure message shows the measured peak.
- Workaround: load weights once, use bf16 or fp16, free caches after warmup, and compare the peak memory practice reports with the limit.
- Status: confirmed in the pinned image

## KSI-009 - Code tried to download something while offline

- Symptom: `LocalEntryNotFoundError`, "offline mode", name resolution or connection errors to huggingface.co, PyTorch hub or timm/torchvision URLs.
- Cause: the policy container has no network.
- Workaround: put everything the policy loads in your Hugging Face snapshot and load it from `/weights` by path. Watch for configs that name hub ids (LeRobot pi0's tokenizer, KSI-024).
- Status: confirmed in the pinned image

## KSI-010 - Missing processor, tokenizer or config files

- Symptom: `Can't load image processor`, `Can't load tokenizer`, or a missing `preprocessor_config.json`.
- Cause: the snapshot lacks processor or tokenizer files.
- Workaround: add them to your snapshot and load by path. For LeRobot, point `tokenizer_name` in `policy_preprocessor.json` at `/weights/...` (KSI-024).
- Status: reported upstream

## KSI-011 - Checkpoint format drift

- Symptom: `Unexpected key(s) in state_dict`, `Missing key(s)`, `size mismatch`, or an unknown model type.
- Cause: the checkpoint was saved with other LeRobot, PEFT or Transformers versions than the image pins.
- Workaround: re-save with the image's versions or convert in your code; load your real snapshot with the practice tool first (KSI-023, KSI-025).
- Status: confirmed in the pinned image

## KSI-013 - Writes outside the scratch directory fail

- Symptom: `Read-only file system` or `[Errno 30]`.
- Cause: only `/scratch` is writable.
- Workaround: point caches and temporary files (`HOME`, `TMPDIR`, `XDG_CACHE_HOME`) at `/scratch`.
- Status: confirmed in the pinned image

## KSI-015 - MuJoCo rendering: use EGL or OSMesa, not GLFW

- Symptom: `MUJOCO_GL=glfw` or a viewer fails with `an OpenGL platform library has not been loaded`; harmless `EGLError` or `ctypes.ArgumentError` tracebacks print at exit.
- Cause: there is no display. The image defaults to `MUJOCO_GL=egl` and also has OSMesa.
- Workaround: keep the default (or `osmesa`); do not use GLFW. Call `renderer.close()` before exit to avoid the tracebacks. Prefer the observation cameras to rendering in the policy.
- Status: confirmed in the pinned image

## KSI-016 - tiktoken needs its BPE files offline

- Symptom: `tiktoken.get_encoding("cl100k_base")` fails to resolve `openaipublic.blob.core.windows.net`.
- Cause: tiktoken downloads its rank file on first use.
- Workaround: ship the rank file in your snapshot and load it with `tiktoken.load.load_tiktoken_bpe("/weights/...")`, or copy it to `/weights/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4` and set `TIKTOKEN_CACHE_DIR=/weights/tiktoken`.
- Status: confirmed in the pinned image

## KSI-017 - gymnasium-robotics Fetch and FrankaKitchen environments fail

- Symptom: `gym.make("FetchReach-v4")` or `FrankaKitchen-v1` raises a bare `AssertionError`.
- Cause: gymnasium-robotics 1.4.2 is incompatible with the image's mujoco 4.0.1.
- Workaround: do not build those environments in the policy; copy the constants you need.
- Status: confirmed in the pinned image

## KSI-018 - jaxtyping with typeguard raises SyntaxError

- Symptom: `@jaxtyped(typechecker=typeguard.typechecked)` with `Float[Tensor, "b c"]` raises `SyntaxError`.
- Cause: typeguard parses annotation strings as code.
- Workaround: use `typechecker=None` or `isinstance` checks.
- Status: confirmed in the pinned image

## KSI-019 - ModuleNotFoundError or ImportError when the policy loads

- Symptom: `weights_load_failed` with `ModuleNotFoundError: No module named 'x'` or `ImportError: cannot import name ...`.
- Cause: nothing is installed from your repository, and a vendored package named like an image package is never used.
- Workaround: read the checks' import warning; vendor pure-Python code under your own top-level name, guard optional imports, or ask the organizers to add a package ([SUBMIT.md](SUBMIT.md)).
- Status: confirmed in the pinned image

## KSI-021 - Program never listens on the policy port

- Symptom: `container_unhealthy` (exit 78) after up to 10 minutes: `nothing is listening on port 8080 after 600 s`, possibly naming another port.
- Cause: your server bound another port (Flask 5000, uvicorn 8000), bound only to `127.0.0.1`/`localhost`, or loaded the model before binding.
- Workaround: listen on `0.0.0.0` and `int(os.environ.get("RC_POLICY_PORT", "8080"))`; start the server first, then load and report `ready: false` until done.
- Status: confirmed in the pinned image

## KSI-022 - import torch at module level failed with reserved_module_shadow (fixed in image v6)

- Symptom: in images v5 and earlier, `reserved_module_shadow` (or exit 72) for any policy importing torch outside `load()`.
- Cause: a runtime loader bug.
- Workaround: none needed from image v6 on.
- Status: fixed in image v6

## KSI-023 - LeRobot checkpoint without processor files: `ProcessorMigrationError`

- Symptom: `ProcessorMigrationError: Model '/weights' requires migration to processor format`.
- Cause: checkpoints saved before LeRobot 0.4 have no `policy_preprocessor.json`; the image's LeRobot refuses them.
- Workaround: run `python -m lerobot.processor.migrate_policy_normalization --pretrained-path <dir>` on your machine, make the files world-readable and upload them.
- Status: confirmed in the pinned image

## KSI-024 - pi0 and pi0.5 need a gated PaliGemma tokenizer

- Symptom: `Failed to instantiate processor step 'tokenizer_processor' ... couldn't connect to 'https://huggingface.co'`.
- Cause: the preprocessor loads the tokenizer from the gated hub id `google/paligemma-3b-pt-224`.
- Workaround: put the tokenizer files in your snapshot (for example `tokenizer/`) and set `tokenizer_name` in `policy_preprocessor.json` to `/weights/tokenizer`. Mind the Gemma licence.
- Status: confirmed in the pinned image

## KSI-025 - OpenVLA remote code does not load on the pinned transformers and timm

- Symptom: `openvla/openvla-7b` with `trust_remote_code=True` fails on missing transformers names, the timm version check, `tie_weights` arguments or a missing `generate`.
- Cause: its remote code targets transformers 4.40 and timm 0.9; the image has transformers 5 and timm 1.0.
- Workaround: patch the `.py` files in your snapshot: import `PaddingStrategy` from `transformers.utils` and the rest from `transformers.tokenization_utils_base`; drop the timm check; give `tie_weights` `*args, **kwargs`; add `GenerationMixin` as a base; load the class with `get_class_from_dynamic_module("modeling_prismatic.OpenVLAForActionPrediction", "/weights")`.
- Status: confirmed in the pinned image

## KSI-026 - `.to(torch.bfloat16)` on a LeRobot policy fails at the first action

- Symptom: `/act` returns 500: `mat1 and mat2 must have the same dtype, but got Float and BFloat16` (SmolVLA, ACT).
- Cause: `select_action` builds some tensors in float32.
- Workaround: keep fp32 weights or run under `torch.autocast("cuda", dtype=torch.bfloat16)`. For pi0/pi0.5 set `config.dtype = torch.bfloat16` (a dtype, not a string).
- Status: confirmed in the pinned image

## KSI-027 - `flash_attn` is not installed (`ModuleNotFoundError` or "FlashAttention2 has been toggled on")

- Symptom: `No module named 'flash_attn'` (or `flash_attn_interface`), or "FlashAttention2 has been toggled on, but it cannot be used".
- Cause: the image ships no flash-attn; the model or its config asks for `flash_attention_2`.
- Workaround: switch to SDPA; see "flash-attention to PyTorch SDPA" in [SUBMIT.md](SUBMIT.md).
- Status: confirmed in the pinned image

## KSI-028 - `lerobot-train` workers die: "unable to allocate shared memory"

- Symptom: training in the image on your machine stops with `unable to allocate shared memory(shm)`.
- Cause: Docker's default `/dev/shm` is 64 MB.
- Workaround: pass `--shm-size=8g` (or `--ipc=host`), or `num_workers=0` ([PRACTICE.md](PRACTICE.md)).
- Status: reported by a practice team

## KSI-029 - Downloads inside the image fail with `PermissionError` on `/hf-base`

- Symptom: `snapshot_download`, `from_pretrained` or `lerobot-train` fails with `PermissionError` when you run the image yourself.
- Cause: the image sets `HF_HUB_CACHE=/hf-base` (read-only), which wins over `HF_HOME`.
- Workaround: for training or export runs pass `-e HF_HUB_CACHE=/scratch/hf`; never set it in your policy ([PRACTICE.md](PRACTICE.md)).
- Status: reported by a practice team

## KSI-030 - LeRobot fine-tuning with different camera names: "Feature mismatch"

- Symptom: fine-tuning `smolvla_base` or `pi05_base` fails with `Feature mismatch ... Missing camera1..3`.
- Cause: `--rename_map` only renames; it cannot add a camera the base config lacks.
- Workaround: pass `--policy.input_features='{...}' --policy.output_features='{...}'` listing every camera, the state and the action.
- Status: reported by a practice team

## KSI-031 - SmolVLA `use_peft=true` does not train the vision tower

- Symptom: far fewer trainable parameters than expected; the VLM stays frozen.
- Cause: the default PEFT targets cover only the action expert.
- Workaround: pass a `--peft.target_modules` regex for the modules you want and check the parameter count.
- Status: reported by a practice team

## KSI-032 - MolmoAct2 offline: processor needs dataset metadata, model needs merged HF shards

- Symptom: building the processors raises `normalize_gripper=False cannot identify the gripper dimension`; loading from a LeRobot-only `model.safetensors` fails.
- Cause: the processor reads dataset feature names; the policy re-reads Hugging Face shards from `checkpoint_path`.
- Workaround: pass `dataset_feature_names` with a gripper entry (or `normalize_gripper=True`), and ship the merged weights as HF shards.
- Status: reported by a practice team

## KSI-033 - `lerobot-train` refuses to restart: `FileExistsError` on the output directory

- Symptom: restarting a job fails with `FileExistsError` on its output directory.
- Cause: LeRobot will not overwrite an existing run.
- Workaround: restart with `--resume=true` and that run's `--output_dir`, or remove the directory.
- Status: reported by a practice team

## KSI-035 - MolmoAct2: smaller frames do not cut latency; each extra image adds prefill time

- Symptom: downscaling frames does not speed up `/act`; adding the goal image slows every call.
- Cause: the processor resizes every image to 378×378 (196 tokens) whatever size you pass.
- Workaround: send fewer images, use fewer flow steps or CUDA graphs, and return action chunks to fit the 90 s `/act` budget ([budget arithmetic](../docs/api.md#act-budget-arithmetic)).
- Status: reported by a practice team

## KSI-036 - LeRobot `multi_task_dit`: "stack expects each tensor to be equal size"

- Symptom: `RuntimeError: stack expects each tensor to be equal size` with the goal image as a camera.
- Cause: the goal image is 640×480; the cameras are 640×360.
- Workaround: resize every image to one size before stacking, in training and in `act()`.
- Status: reported by a practice team
