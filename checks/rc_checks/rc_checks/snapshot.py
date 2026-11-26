"""Checks on a downloaded Hugging Face snapshot directory (native binaries, unsafe entries)."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Optional

from . import _native
from .findings import CheckResult, make
from .release import Release


MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIGS = 200
# Formats the image cannot run, or that need care, by extension (warn only). Order is display order.
FORMAT_NOTES = {
    ".onnx": "ONNX models run through onnxruntime-gpu; its CUDA provider loads only after `import torch` or `onnxruntime.preload_dlls()`, otherwise it silently falls back to the CPU. There is no onnx package. Converting to safetensors and running in torch is the most tested path.",
    ".gguf": "GGUF needs llama.cpp, which is not in the image; convert the model to safetensors.",
    ".tflite": "TFLite needs TensorFlow, which is not in the image; convert the model to safetensors or JAX/orbax.",
    ".h5": "Keras HDF5 files load with keras and h5py (both installed; KERAS_BACKEND defaults to jax), but TensorFlow is not installed, so models saved with the TensorFlow backend or TF-only layers can fail. Safetensors is safer.",
    ".hdf5": "Keras HDF5 files load with keras and h5py (both installed; KERAS_BACKEND defaults to jax), but TensorFlow is not installed, so models saved with the TensorFlow backend or TF-only layers can fail. Safetensors is safer.",
    ".keras": "Keras files load with keras (installed; KERAS_BACKEND defaults to jax, torch also works), but TensorFlow is not installed, so TF-only layers or models saved with the TensorFlow backend can fail. Safetensors is safer.",
}
SAVED_MODEL_NOTE = "TensorFlow SavedModel needs TensorFlow, which is not in the image; convert the model to safetensors or JAX/orbax."
ADAPTER_NOTE = ("adapter_config.json names the base model {base!r}, which is not in this snapshot and not on the curated "
                "base-model list. There is no network at judging, so from_pretrained({base!r}) cannot download it. Merge the "
                "adapter into the base and upload the merged model, or include the base model in this repository and load it "
                "by path (then ignore this note).")
QUANT_UNSUPPORTED = {
    "gptq": "GPTQ quantization needs auto-gptq/optimum, which are not installed; upload BF16 weights (merged, not quantized).",
    "awq": "AWQ quantization needs autoawq, which is not installed; upload BF16 weights (merged, not quantized).",
}


def _read_json(path: Path):
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _curated_bases(release: Optional[Release]) -> set:
    cache = (release.raw.get("base_cache") if release else None) or {}
    entries = cache.get("entries") if isinstance(cache, dict) else None
    return {e.get("repo_id") for e in entries or [] if isinstance(e, dict) and isinstance(e.get("repo_id"), str)}


def _config_findings(root: Path, rel: str, path: Path, curated: set, result: CheckResult) -> None:
    name = path.name
    data = _read_json(path)
    if not isinstance(data, dict):
        return
    if name == "adapter_config.json":
        base = data.get("base_model_name_or_path")
        if isinstance(base, str) and base and base not in curated and not os.path.isabs(base) \
                and not base.startswith(".") and not (root / base).exists():
            result.findings.append(make("snapshot_format", rel, detail=ADAPTER_NOTE.format(base=base)))
    elif name == "config.json":
        quant = data.get("quantization_config")
        if isinstance(quant, dict):
            method = str(quant.get("quant_method") or ("bitsandbytes" if quant.get("load_in_4bit") or quant.get("load_in_8bit") else "")).lower()
            if method in QUANT_UNSUPPORTED:
                result.findings.append(make("snapshot_format", rel, detail=QUANT_UNSUPPORTED[method]))
            elif method and method != "bitsandbytes":
                result.findings.append(make("snapshot_format", rel, detail=(
                    f"quantization method {method!r} is not installed or not tested on the judged H100 image; "
                    "only bitsandbytes is available. Upload BF16 weights (merged, not quantized).")))


def check_snapshot(snapshot_dir: Path, *, release: Optional[Release] = None) -> CheckResult:
    root = Path(snapshot_dir)
    result = CheckResult()
    count = 0
    curated = _curated_bases(release)
    formats: dict = {}  # extension or name -> [rel]
    configs = 0
    for current, dirs, files in os.walk(root):
        rel_dir = Path(current).relative_to(root)
        if rel_dir == Path(".") and ".cache" in dirs:  # huggingface_hub local_dir bookkeeping
            dirs.remove(".cache")
        dirs.sort()
        for name in list(dirs):
            if os.path.islink(Path(current) / name):
                dirs.remove(name)
                result.findings.append(make("snapshot_unsafe_entry", (rel_dir / name).as_posix(), why="a symbolic link"))
        for name in sorted(files):
            rel = (rel_dir / name).as_posix()
            path = Path(current) / name
            posix = PurePosixPath(rel)
            if path.is_symlink():
                result.findings.append(make("snapshot_unsafe_entry", rel, why="a symbolic link"))
                continue
            if posix.is_absolute() or ".." in posix.parts:
                result.findings.append(make("snapshot_unsafe_entry", rel, why="outside the snapshot"))
                continue
            count += 1
            try:
                size = path.stat().st_size
            except OSError:
                continue
            plain = _native.extension(name)
            if plain in _native.CUDA_SOURCE_EXTS:
                result.findings.append(make("snapshot_cuda_source", rel))
                continue
            ext = _native.native_extension(name)
            reason = f"extension {ext}" if ext else _native.magic_reason(path, size)
            if not reason and plain in _native.ENGINE_EXTS:
                reason = f"extension {plain}; {_native.ENGINE_REASON}"
            if not reason and plain in _native.SCANNED_ZIP_EXTS:
                reason = _native.zip_native_reason(path)
            if reason:
                result.findings.append(make("snapshot_native_binary", rel, reason=reason))
                continue
            if plain in _native.ARCHIVE_EXTS and plain not in _native.ARCHIVE_EXEMPT | _native.CHECKPOINT_EXTS:
                result.findings.append(make("snapshot_archive", rel, ext=plain))
                continue
            if plain in FORMAT_NOTES:
                formats.setdefault(plain, []).append(rel)
            elif name == "saved_model.pb":
                formats.setdefault(name, []).append(rel)
            elif name in ("adapter_config.json", "config.json") and configs < MAX_CONFIGS:
                configs += 1
                _config_findings(root, rel, path, curated, result)
    for key, rels in formats.items():
        note = SAVED_MODEL_NOTE if key == "saved_model.pb" else FORMAT_NOTES[key]
        more = f" ({len(rels) - 1} more file(s) with the same format)" if len(rels) > 1 else ""
        result.findings.append(make("snapshot_format", rels[0], detail=note + more))
    if count == 0 and not result.findings:
        result.findings.append(make("snapshot_empty"))
    return result
