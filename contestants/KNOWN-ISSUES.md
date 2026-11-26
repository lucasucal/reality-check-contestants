# What's in the image

Last updated: 2026-11-26

What the judged policy image (NVIDIA H100, amd64, Python 3.12.3, CUDA 13) gives your policy, and the rules it runs under. When a failed run matches one of the KSI entries below, your submission page links to it (not for failures during the hidden tasks). IDs are stable and never reused.

## Packages

| Area | Packages |
|---|---|
| PyTorch | torch 2.14.1, torchvision 0.29.1, triton 3.8.0 |
| Models | transformers 5.22.2, lerobot 0.7.0, diffusers 0.39.0, accelerate 1.16.0, peft 0.22.0, timm 1.0.31, safetensors 0.8.0, huggingface-hub 1.33.0, bitsandbytes 0.50.2, einops 0.8.2 |
| JAX | jax 0.12.1, flax 0.12.12, keras 3.15.1 (`KERAS_BACKEND=jax`) |
| ONNX | onnxruntime-gpu 1.32.0 |
| Robotics and vision | mujoco 4.0.1 (`MUJOCO_GL=egl`), mink 1.4.0, ikpy 4.5.0, opencv-contrib-python-headless 4.13.0.92, open3d 0.20.0, pillow 13.0.0 |
| Numerics | numpy 2.2.6, scipy 1.18.1, scikit-learn 1.9.1 |
| Serving | fastapi 0.165.0, uvicorn 0.54.0 |

Every package and version: [`runtime/locks/requirements.amd64.txt`](../runtime/locks/requirements.amd64.txt) (judging). Practice on arm64 uses [`requirements.arm64.txt`](../runtime/locks/requirements.arm64.txt), with the same versions.

Not included: FlashAttention and TensorFlow (below), and anything else not in the lock file.

## KSI-003 - FlashAttention is not in the image: use SDPA

- Symptom: `No module named 'flash_attn'`, or "FlashAttention2 has been toggled on, but it cannot be used".
- Workaround: `attn_implementation="sdpa"` (also in `config.json`); see [flash-attention to PyTorch SDPA](SUBMIT.md#flash-attention-to-pytorch-sdpa).

## KSI-006 - TensorFlow is not in the image

- Symptom: `No module named 'tensorflow'`.
- Workaround: convert the model to PyTorch or JAX; Keras 3 runs on JAX or torch.

## KSI-009 - Offline: load everything from `/weights`

- Symptom: `LocalEntryNotFoundError`, "offline mode", `Can't load tokenizer`, or connection errors to huggingface.co or other hosts.
- Workaround: the policy has no network. Put every file it loads in your one model repo and load it by path from `/weights`, including configs that name a hub id ([your model](SUBMIT.md#your-model)).

## KSI-013 - `/scratch` is the only writable path

- Symptom: `Read-only file system` or `[Errno 30]`.
- Workaround: write caches and temporary files under `/scratch` (the image already points `HOME`, `TMPDIR`, `HF_HOME` and the compile caches there). It is empty at every start.

## KSI-021 - Listen on the policy port before loading the model

- Symptom: `container_unhealthy` (exit 78, `nothing is listening on port 8080`) or `warmup_timeout`.
- Workaround: a program entry point binds `0.0.0.0:$RC_POLICY_PORT` (8080) first, then loads, and reports `ready: false` on `/readyz` until loaded (the class form does this for you). Load plus warmup must fit the startup limit ([limits](RULES.md#limits)).

## KSI-019 - Nothing is installed at judging time

- Symptom: `ModuleNotFoundError` or `ImportError: cannot import name ...` when the policy loads.
- Workaround: use the image's packages plus pure-Python code in your repo, vendored under your own top-level name ([what is installed](SUBMIT.md#you-submit-python-source-only-nothing-is-installed)).

## KSI-008 - One combined CPU + GPU memory limit

- Symptom: `policy_memory_exceeded`, `CUDA out of memory`, or exit code 137.
- Workaround: compare the peak memory practice reports with the [limit](RULES.md#limits); load weights once, in bf16 or fp16.

## Problems with a specific model

Open an issue on the contestant pack repo: <https://github.com/lucasucal/reality-check-contestants/issues>.
