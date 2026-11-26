"""Pinned inputs: exact Git commit, Hugging Face snapshot SHA, the provenance triple,
and the practice-time snapshot advisories (section 10). Network access is injectable."""
from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

SHA40 = re.compile(r"^[0-9a-f]{40}$")
HF_URL = re.compile(r"^https?://(?:www\.)?huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:/tree/([^/]+))?/?$")
PRACTICE_LABEL = ("PRACTICE RESULT: functional check on {where}. It is not proof of judged H100 behavior, timing or memory, "
                  "and not a hidden-task score. Practice is optional and never a submission prerequisite.")


class PinError(Exception):
    """Cannot establish an exact pinned input; the message is team-facing."""


@dataclass(frozen=True)
class GitState:
    sha: str
    branch: Optional[str]          # None when HEAD is detached
    dirty_paths: tuple

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_paths)


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PinError(f"cannot run git in {repo}: {exc}") from exc
    if out.returncode != 0:
        raise PinError(f"git {' '.join(args)} failed in {repo}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def resolve_git(repo: Path) -> GitState:
    """HEAD commit SHA plus every modified, staged or untracked path (a dirty tree is not the commit)."""
    sha = _git(repo, "rev-parse", "HEAD").strip()
    if not SHA40.match(sha):
        raise PinError(f"unexpected git HEAD {sha!r}")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    dirty = tuple(line[3:] for line in status.splitlines() if line.strip())
    return GitState(sha, None if branch == "HEAD" else branch, dirty)


def git_warnings(state: GitState, *, allow_dirty: bool) -> list:
    """Team-facing lines. A dirty tree is refused by the caller unless allow_dirty."""
    out = []
    if state.dirty:
        shown = ", ".join(state.dirty_paths[:5]) + (f" and {len(state.dirty_paths) - 5} more" if len(state.dirty_paths) > 5 else "")
        out.append(("DIRTY WORKING TREE: " + shown + ". Practice runs the committed content of "
                    f"{state.sha[:12]}, NOT your uncommitted changes, and a submission also uses the commit. "
                    "Commit (and push) first." + (" Continuing because --allow-dirty was given." if allow_dirty else "")))
    if state.branch:
        out.append(f"HEAD is on branch {state.branch!r}, a moving ref; submissions pin the commit it resolves to when you submit.")
    return out


def export_commit(repo: Path, sha: str, dest: Path) -> Path:
    """Materialize the committed tree (no .git) like the judged clone: `git archive <sha>`."""
    import tarfile
    dest.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", sha], stdout=subprocess.PIPE)
    except OSError as exc:
        raise PinError(f"cannot run git archive: {exc}") from exc
    assert proc.stdout is not None
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        tar.extractall(dest, filter="tar")
    if proc.wait() != 0:
        raise PinError(f"git archive {sha} failed")
    return dest


@dataclass(frozen=True)
class HFRef:
    repo_id: str
    requested: Optional[str]       # revision as the team gave it (None = default branch)


def parse_model(value: str, revision: Optional[str] = None) -> HFRef:
    """Accepts https://huggingface.co/org/name[/tree/rev] or a bare org/name."""
    value = value.strip()
    m = HF_URL.match(value)
    if m:
        return HFRef(m.group(1), revision or m.group(2))
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        return HFRef(value, revision)
    raise PinError(f"{value!r} is not a Hugging Face model link (expected https://huggingface.co/<org>/<name>)")


Fetch = Callable[[str], dict]


def _fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https host
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise PinError(f"Hugging Face returned HTTP {exc.code} for {url} (is the model public and ungated, and the revision real?)") from exc
    except (OSError, ValueError) as exc:
        raise PinError(f"cannot reach Hugging Face ({url}): {exc}") from exc


@dataclass(frozen=True)
class HFSnapshot:
    repo_id: str
    sha: str
    warnings: tuple = field(default_factory=tuple)


def resolve_hf(ref: HFRef, fetch: Fetch = _fetch_json) -> HFSnapshot:
    """Resolve to the exact snapshot SHA; warn on a moving revision or a newer default revision."""
    api = f"https://huggingface.co/api/models/{ref.repo_id}"
    wanted = ref.requested or "main"
    info = fetch(f"{api}/revision/{wanted}")
    sha = info.get("sha")
    if not isinstance(sha, str) or not SHA40.match(sha):
        raise PinError(f"Hugging Face gave no snapshot SHA for {ref.repo_id}@{wanted}")
    warnings = []
    if not SHA40.match(wanted):
        warnings.append(f"revision {wanted!r} is a moving ref; practice uses snapshot {sha}. A submission pins the snapshot current "
                        "at submit time unless you give an exact SHA.")
    else:
        latest = fetch(f"{api}/revision/main").get("sha")
        if isinstance(latest, str) and latest != sha:
            warnings.append(f"the default revision of {ref.repo_id} is newer ({latest}) than the one you practiced with ({sha}); "
                            "a submission without a revision uses the newer one.")
    return HFSnapshot(ref.repo_id, sha, tuple(warnings))


def _looks_like_transformers_config(path: Path) -> bool:
    """True when config.json is a Hugging Face transformers/processor config (has `model_type` or `architectures`).
    A plain MLP or other framework config, or unparseable JSON, is not."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and ("model_type" in data or "architectures" in data)


def snapshot_warnings(snapshot_dir: Path) -> list:
    """Advisories on a downloaded snapshot: nested checkpoint, missing processor/tokenizer files."""
    root = Path(snapshot_dir)
    out = []
    top_weights = [p for p in root.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt")]
    nested = sorted({p.parent.relative_to(root).as_posix() for p in root.rglob("*.safetensors") if p.parent != root})
    if not top_weights and nested:
        out.append(f"the checkpoint files are in a subfolder ({', '.join(nested[:3])}), not the snapshot root; your policy must load "
                   "from /weights/<subfolder> (multi-model snapshots are fine, a forgotten nesting is not).")
    has_config = _looks_like_transformers_config(root / "config.json")
    names = {p.name for p in root.iterdir()}
    if has_config and not (names & {"tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                                   "processor_config.json", "policy_preprocessor.json"}):
        out.append("config.json is present but no tokenizer or processor files were found; if your policy builds a processor "
                   "from the hub id, it cannot load offline: add those files to your snapshot.")
    return out


def provenance_block(*, git_sha: str, hf_repo: Optional[str], hf_sha: Optional[str], image_platform_digest: str,
                     where: str, manifest_digest: Optional[str] = None, tree_sha256: Optional[str] = None,
                     weights_dir: Optional[str] = None) -> str:
    """hf_repo None: local weights from weights_dir, no model id. With both, the weights are not checked against the snapshot."""
    lines = ["Provenance (what this practice run used)",
             f"  Git commit:             {git_sha}",
             f"  HF snapshot:            {f'{hf_repo}@{hf_sha}' if hf_repo else 'none (local weights, no model id)'}"]
    if weights_dir:
        lines.append(f"  Weights:                {weights_dir} (--weights-dir"
                     f"{', not checked against the HF snapshot' if hf_repo else ''})")
    lines.append(f"  Image platform digest:  {image_platform_digest}")
    if manifest_digest:
        lines.append(f"  Image manifest digest:  {manifest_digest}")
    if tree_sha256:
        lines.append(f"  Submission tree hash:   {tree_sha256}")
    lines.append(PRACTICE_LABEL.format(where=where))
    return "\n".join(lines)
