"""Finding and CheckResult types plus the stable message table.

Every team-facing message is rendered by :func:`render` only, so the backend
and the practice launcher print byte-identical text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Severity = Literal["reject", "warn"]

CAVEAT = (
    " This check reads file headers only; it cannot see dtype casts or quantization applied "
    "when your code loads the model (for example load_in_4bit). See {docs}/numeric-formats. "
    "FP32, BF16 and FP16 are the baseline; consider merging adapters onto a BF16 base."
)

# code -> (severity, template)
MESSAGES: dict = {
    # repo, reject
    "manifest_missing": ("reject", "Add manifest.json at the repository root."),
    "manifest_invalid": ("reject", "manifest.json must be a strict JSON object (no comments or trailing commas): {detail}"),
    "manifest_contract": ("reject", 'manifest.json "contract" must be exactly "v0".'),
    "manifest_policy_schema": ("reject", 'Class policies must set "schema_version": {version} in manifest.json. Update your policy to use the current observation and action contract.'),
    "manifest_port": ("reject", 'manifest.json "port" must be 8080 or omitted; the runtime image always serves on 8080.'),
    "context_no_dockerfile": ("reject", "Add a Dockerfile at the repository root."),
    "weights_in_repo": ("reject", "Remove the top-level weights/ directory; weights belong on Hugging Face."),
    "context_submodules": ("reject", "Git submodules are not supported; include their files in the repository."),
    "context_lfs": ("reject", "{path} is a Git LFS pointer; Git LFS is not supported, include the actual file (or put it in your Hugging Face model)."),
    "context_too_heavy": ("reject", "The repository exceeds the {limit} limit ({actual}). Move large files to your Hugging Face model."),
    "repo_native_binary": ("reject", "{path} looks like a compiled native binary ({reason}). Only pure-Python code is allowed in the repository; the judged image provides all packages."),
    "repo_cuda_source": ("reject", "{path} is CUDA source (not compiled or used; custom CUDA extensions are not supported). Remove it; only pure-Python code is allowed in the repository and the judged image provides all packages."),
    "repo_archive": ("reject", "{path} is an archive ({ext}); archives are not allowed because their contents cannot be checked. Unpack it into plain files. (.npz, .pt, .pth and .safetensors are allowed.)"),
    "repo_symlink": ("reject", "{path} is a symbolic link; symbolic links are not allowed. Replace it with a real file."),
    "repo_reserved_name": ("reject", "{path} would run at interpreter start-up and is not allowed at the repository root. Remove it."),
    "entry_point_missing": ("reject", 'manifest.json needs "entry_point": "package.module:Attr" (a Policy class), or a Python program that serves the contract: a dotted module ("mypkg.server") or a relative .py path ("serve.py").'),
    "entry_point_invalid": ("reject", 'entry_point {value!r} is not valid: {reason}. Use "package.module:Attr" (a Policy class), or a Python program that serves the contract: a dotted module ("mypkg.server") or a relative .py path ("serve.py").'),
    "entry_module_not_found": ("reject", "entry_point module {module} not found; expected {module_path}.py or {module_path}/__init__.py in the repository."),
    "entry_program_not_found": ("reject", "entry_point {value!r} not found in the repository; expected {expected}."),
    "entry_syntax_error": ("reject", "{path} line {line}: Python syntax error: {detail}"),
    "entry_attr_missing": ("reject", "{path} does not define {attr} at module top level."),
    "entry_shadowed": ("reject", "entry_point top-level package {name!r} collides with a module that already exists in the judged image ({kind}); the image's module would be loaded instead of yours. Rename your package."),
    # repo, warn
    "entry_syntax_unverified": ("warn", "{path} line {line}: this checker's Python is not the judged image's Python 3.12, so it cannot be sure this is a syntax error: {detail}. The policy fails to load if it really is invalid under Python 3.12."),
    "dockerfile_ignored": ("warn", "Your Dockerfile is not used for judging. We run only your Python code inside the official policy image; changes to the Dockerfile, base image, installed packages, SDK or entrypoint are ignored."),
    "repo_bytecode_present": ("warn", "Compiled Python files ({paths}) were removed; only your .py sources are mounted."),
    "repo_runtime_copy_ignored": ("warn", "The image provides rc_sdk/rc_runtime; your copy at {path} is ignored."),
    "repo_native_source": ("warn", "{path} is native source or build configuration; nothing is compiled, so it will have no effect."),
    "repo_module_shadowed": ("warn", "Imports of {name!r} will resolve to the judged image's {name!r}, not your file {path}; rename it if your code imports it."),
    "manifest_unknown_key": ("warn", "manifest.json key {key!r} is ignored."),
    "import_unresolved": ("warn", "Your code imports modules that are neither installed in the judged image nor provided by your repository: {modules}. Nothing is installed from your repository, so an unguarded import fails when the policy loads (ModuleNotFoundError, known issue KSI-019).{alternatives} See the supported-package list in contestants/SUBMIT.md (section \"You submit Python source only; nothing is installed\"); copy pure-Python code into your repository under your own name, guard optional imports with try/except ImportError, or ask the organizers to add a package."),
    "dependencies_ignored": ("warn", "{files} declare dependencies, but nothing is ever installed from your repository: the judged image has a fixed package set and these files are ignored. Check your imports against the supported-package list in contestants/SUBMIT.md (section \"You submit Python source only; nothing is installed\"); to get a package added, ask the organizers."),
    "hub_id_offline": ("warn", "Your code loads Hugging Face hub ids by name: {ids}. There is no network when your policy loads (HF_HUB_OFFLINE=1) and no organizer-provided base models, so loading by hub id fails at load with an OSError. Upload the files to your model repository and load them from /weights by path; see \"Your model\" in contestants/SUBMIT.md."),
    "flash_attention_requested": ("warn", "Your code requests flash-attention: {uses}. flash-attn is not installed in the judged image, so the model fails to load (ImportError). Use attn_implementation=\"sdpa\" (same results); see \"flash-attention to PyTorch SDPA\" in contestants/SUBMIT.md."),
    "program_binds_loopback": ("warn", "{path} line {line}: your program binds its server to {host!r}. The runner connects over the container network, so a server listening only on {host} is never reached and startup fails when the listen deadline passes. Listen on 0.0.0.0 and the port in RC_POLICY_PORT (8080); see \"Repository layout\" in contestants/SUBMIT.md."),
    # snapshot
    "snapshot_native_binary": ("reject", "Model snapshot file {path} looks like a compiled native binary ({reason}). Native binaries are not allowed in the model repository; remove it or use a clean revision without it."),
    "snapshot_cuda_source": ("reject", "Model snapshot file {path} is CUDA source (not compiled or used; custom CUDA extensions are not supported). Remove it from the model repository or use a clean revision without it."),
    "snapshot_archive": ("reject", "Model snapshot file {path} is an archive ({ext}); archives are not allowed because their contents cannot be checked. Unpack it into plain files. (.npz, .pt, .pth, .bin, .ckpt, .npy, .gguf and .safetensors are allowed.)"),
    "snapshot_unsafe_entry": ("reject", "Model snapshot entry {path} is {why} (symbolic link or path outside the snapshot); remove it from the model repository."),
    "snapshot_empty": ("reject", "The model snapshot contains no files."),
    "snapshot_format": ("warn", "Model snapshot file {path}: {detail}"),
    # dtype
    "dtype_non_baseline": ("warn", "{path} contains tensors in formats that are not tested on the judged RTX PRO 6000 image: {formats}.{caveat}"),
    "dtype_unchecked_format": ("warn", "{n} file(s) with extension {ext} ({example}...) cannot be dtype-checked cheaply; numeric format is your responsibility.{caveat}"),
    "dtype_header_unreadable": ("warn", "Could not read the safetensors header of {path}; numeric format was not checked.{caveat}"),
}


def severity_of(code: str) -> str:
    return MESSAGES[code][0]


def render(code: str, **fields) -> str:
    """Render the team-facing message for ``code``. ``docs`` feeds the dtype caveat."""
    template = MESSAGES[code][1]
    if "{caveat}" in template:
        fields["caveat"] = CAVEAT.format(docs=fields.get("docs", ""))
    return template.format(**fields)


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    path: Optional[str] = None
    line: Optional[int] = None

    def to_dict(self) -> dict:
        out = {"code": self.code, "severity": self.severity, "path": self.path, "message": self.message}
        if self.line is not None:
            out["line"] = self.line
        return out


def make(code: str, path: Optional[str] = None, line: Optional[int] = None, **fields) -> Finding:
    """Build a Finding from the message table (severity and text come from ``code``)."""
    return Finding(code, severity_of(code), render(code, path=path, line=line, **fields), path, line)


@dataclass
class CheckResult:
    findings: list = field(default_factory=list)

    @property
    def rejects(self) -> list:
        return [f for f in self.findings if f.severity == "reject"]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.rejects

    def extend(self, other: "CheckResult") -> None:
        self.findings.extend(other.findings)

    def to_dict(self) -> dict:
        ordered = sorted(self.findings, key=lambda f: (f.severity != "reject", f.code, f.path or ""))
        return {"ok": self.ok, "findings": [f.to_dict() for f in ordered]}
