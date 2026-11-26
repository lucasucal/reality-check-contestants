"""python -m practice run ...   (run from the participant bundle root)

  python -m practice run --repo /path/to/policy --model https://huggingface.co/org/name [--revision SHA]
  python -m practice run --repo /path/to/policy --weights-dir DIR      (local weights, no model id)

Human-readable output goes to stderr; stdout carries one JSON object so it can be piped: the score
(`scorer`, `total`, `breakdown`), `provenance`, `label`, and the rows of rc_runner.practice.report
(`episodes`, `startup`, `act_budget_s`). Exit codes: 0 scored, 1 refused or rejected by the shared
checks (as a submission would be), 2 setup, engine or policy failure.
"""
from __future__ import annotations

import argparse
import json
import platform as _platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import rc_checks
from rc_runner import launch
from rc_runner.docker_driver import DockerError

from . import checks_run, hints, launcher, provenance, resources
from .provenance import PinError


def say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def say_failure(text: str, verdict: str, engine_url: str = "http://127.0.0.1:8090") -> None:
    """An error line, the one next step when the error is a common setup problem, and the VERDICT line."""
    say(text)
    step = hints.next_step(text, engine_url)
    if step:
        say(f"Next step: {step}")
    say(f"VERDICT: {verdict}")


@dataclass
class Prepared:
    git: provenance.GitState
    hf: Optional[provenance.HFSnapshot]  # None: --weights-dir without --model (local weights, no model id)
    tree: Path
    weights: Path
    release: "rc_checks.Release"
    outcome: checks_run.CheckOutcome
    tree_sha256: str
    workdir: Path
    notes: list = field(default_factory=list)
    release_source: str = "release.json"


@dataclass
class Deps:
    """Everything that touches the network or the machine, replaceable in tests."""
    fetch: provenance.Fetch = provenance._fetch_json
    download: Optional[Callable[[str, str, Path], None]] = None
    make_driver: Optional[Callable[[], Any]] = None


def hf_download(repo_id: str, sha: str, dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PinError("downloading a model needs huggingface_hub: in runner/ run `uv sync --group practice` "
                       "(or `pip install huggingface_hub`), or pass --weights-dir with the snapshot") from exc
    snapshot_download(repo_id, revision=sha, local_dir=str(dest))


def prepare(args: argparse.Namespace, deps: Deps, *, say: Callable[[str], None] = say) -> Prepared:
    """Resolve pinned inputs and run the shared checks. Raises PinError when practice must not proceed."""
    repo = Path(args.repo).resolve()
    git = provenance.resolve_git(repo)
    for line in provenance.git_warnings(git, allow_dirty=args.allow_dirty):
        say("WARNING: " + line)
    if git.dirty and not args.allow_dirty:
        raise PinError("the working tree has uncommitted changes, so what you would practice is not what you would submit. "
                       "Next step: commit them (git add -A && git commit -m wip), or pass --allow-dirty to practice the "
                       "last commit anyway.")
    if args.weights_dir and not Path(args.weights_dir).is_dir():
        raise PinError(f"--weights-dir {args.weights_dir} is not a directory. Next step: point it at your checkpoint "
                       "folder, or create the example one with python3 starter/scripts/make_example_weights.py "
                       f"{args.weights_dir}")
    hf: Optional[provenance.HFSnapshot] = None
    if args.model:
        hf = provenance.resolve_hf(provenance.parse_model(args.model, args.revision), deps.fetch)
        for line in hf.warnings:
            say("WARNING: " + line)
        if args.weights_dir:
            say(f"WARNING: the weights come from --weights-dir {args.weights_dir}, which practice does not check against "
                f"{hf.repo_id}@{hf.sha}; a submission uses that snapshot.")
    elif args.weights_dir:
        say("no --model: practicing the local weights in --weights-dir; provenance records no model id. "
            "A submission needs a public Hugging Face model link.")
    else:
        raise PinError("pass --model (your Hugging Face model link) or --weights-dir (a local checkpoint, for example "
                       "the one starter/scripts/make_example_weights.py writes).")
    release_path = Path(args.release) if args.release else rc_checks.release.default_release_path()
    release = rc_checks.load_release(release_path)
    if release_path == rc_checks.release.FALLBACK_PATH:
        say(f"WARNING: no release manifest found (runtime/release.json, --release, RC_RELEASE_JSON); using the built-in "
            f"package list, which names no policy image. {launcher.RELEASE_HELP}")
    else:
        digest = (release.raw.get("image") or {}).get("manifest_digest") if isinstance(release.raw.get("image"), dict) else None
        # CS-B-11: the same release id and manifest digest the judged submission page shows under "Runtime release".
        say(f"release manifest: {release_path} ({release.release_id or 'no release_id'}"
            f"{f', image manifest {digest}' if digest else ''})")
    workdir = Path(args.work_dir or tempfile.mkdtemp(prefix="rc-practice-"))
    workdir.mkdir(parents=True, exist_ok=True)
    tree = provenance.export_commit(repo, git.sha, workdir / "submission")
    if args.weights_dir:
        weights = Path(args.weights_dir).resolve()
    else:
        weights = workdir / "weights" / hf.sha
        if not weights.is_dir() or not any(weights.iterdir()):
            say(f"downloading {hf.repo_id}@{hf.sha} (network is allowed here; the policy container itself runs offline)")
            (deps.download or hf_download)(hf.repo_id, hf.sha, weights)
    for line in provenance.snapshot_warnings(weights):
        say("WARNING: " + line)
    outcome = checks_run.run_checks(tree, weights, release)
    for line in outcome.lines():
        say(line)
    launch.finalize_submission_tree(tree)  # same strip + a+rX as the judged worker's /submission
    return Prepared(git, hf, tree, weights, release, outcome, rc_checks.tree_sha256(tree), workdir,
                    release_source=str(release_path))


def _score_json(score: dict, prov: dict, label: str, failure: Optional[str] = None) -> str:
    return json.dumps({**score, "provenance": prov, "label": label, **({"error": failure} if failure else {})})


def episode_settings(args: argparse.Namespace) -> Any:
    """--judged: the judged /act budget (--scenario-seconds) and 60 s cap; --max-steps caps either mode."""
    from rc_runner.episode import EpisodeSettings
    if args.judged:
        return EpisodeSettings(episode_timeout=args.scenario_seconds, judged=True, max_steps=args.max_steps)
    return EpisodeSettings(episode_timeout=900.0, max_steps=args.max_steps)


def _episodes(args: argparse.Namespace, engine: Any, policy: Any, alive: Optional[Callable[[], bool]],
              started: Optional[float] = None) -> tuple:
    """Startup/warmup then public-task episodes via rc_runner.practice.
    Returns (score, failure, results, startup timings; see rc_runner.practice.startup_and_warmup)."""
    from rc_runner import practice as rp
    seeds = rp.parse_seeds(args.seeds)
    rp.wait_for_engine(engine, args.ready_timeout, args.engine)
    _, tasks = engine.tasks()
    wanted = [t for t in args.tasks.split(",") if t]
    task_ids = wanted or sorted(t["task_id"] for t in tasks)
    startup = rp.startup_and_warmup(engine, policy, task_id=task_ids[0], acts=args.warmup_acts,
                                    ready_timeout=args.policy_ready_timeout, alive=alive, started=started)
    settings = episode_settings(args)
    score, results, failure = rp.practice(engine, policy, tasks_filter=wanted or None, seeds=seeds,
                                          journal_dir=Path(args.journal_dir) if args.journal_dir else None,
                                          settings=settings)
    return score, failure, results, startup


class LocalDocker:
    """Runs the policy container on this machine's Docker with the judged launch arguments.

    The practice run talks to the policy only through this object, so another host can stand in for it
    (`start`, `startup_log`, `memory_line`, `close` and the attributes below)."""

    platform: Optional[str] = None          # image platform; None = this machine's
    transport_note: Optional[str] = None    # added to the /act latency line
    errors: tuple = ()                      # extra exception types that mean "practice could not run"

    def __init__(self, args: argparse.Namespace, deps: Deps, limits: Any) -> None:
        self.args, self.deps, self.limits = args, deps, limits
        self.driver: Any = None
        self.name = ""
        self.sampler: Optional[resources.MemorySampler] = None
        self.memory_limit = limits.policy_memory

    def start(self, prep: Prepared, image: launcher.ImageRef, prov: dict) -> tuple:
        """Start the policy; returns (policy client, alive() or None, where-label)."""
        from rc_runner.docker_driver import DockerDriver
        args, limits = self.args, self.limits
        self.name = f"rc-practice-policy-{prep.git.sha[:8]}"
        self.driver = driver = self.deps.make_driver() if self.deps.make_driver else DockerDriver(limits.docker_bin)
        settings = launcher.PolicySettings.from_config(limits, gpus=args.gpus, memory=args.memory)
        where = f"{_platform.machine()} host, {launcher.gpu_note(settings)}, image {image.platform}"
        if args.print_argv:
            say("docker run -d " + " ".join(launcher.policy_launch_args(
                driver, name=self.name, image_ref=image.run_ref, submission_dir=str(prep.tree),
                weights_dir=str(prep.weights), settings=settings)))
        launcher.ensure_image(driver, image, pull=not args.no_pull)
        ip = launcher.start_policy(driver, name=self.name, image_ref=image.run_ref, submission_dir=str(prep.tree),
                                   weights_dir=str(prep.weights), settings=settings)
        self.memory_limit = settings.memory
        if not self.deps.make_driver:  # a real Docker host: measure (never limit) the container's memory
            self.sampler = resources.MemorySampler(self.name, gpu=bool(launch.gpu_args(settings.gpus, settings.extra_args)),
                                                   docker_bin=limits.docker_bin).start()
        info = driver.inspect_run_image(self.name, expected_manifest_digest=image.manifest_digest)
        if info.get("platform_digest") != image.platform_digest:
            raise launcher.LaunchError(f"the container runs {info.get('platform_digest')}, not the release's "
                                       f"{image.platform_digest}")
        prov["image_platform_digest"] = info["platform_digest"]
        name = self.name
        policy = resources.TimedPolicyClient(f"http://{ip}:{limits.policy_port}", act_timeout=args.act_timeout,
                                             connect_timeout=5.0, retries=1,
                                             container_alive=lambda: driver.is_running(name))
        return policy, (lambda: driver.is_running(name)), where

    def startup_log(self) -> str:
        return self.driver.logs_tail(self.name, 60) if self.driver is not None else ""

    def memory_line(self) -> str:
        if self.sampler is None:
            return "peak memory: not measured"
        self.sampler.stop()
        return self.sampler.describe()

    def close(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()
        if self.driver is not None:
            self.driver.remove(self.name)


def run_command(args: argparse.Namespace, deps: Optional[Deps] = None,
                host_factory: Optional[Callable[[argparse.Namespace, Deps, Any], Any]] = None) -> int:
    from rc_runner.engine_client import EngineClient
    from rc_runner.errors import EngineInfraError
    from rc_runner.policy_limits import PolicyLimits
    from rc_runner import practice as rp
    deps = deps or Deps()
    say("Reality Check practice: optional, never a submission prerequisite.")
    try:
        prep = prepare(args, deps)
    except (PinError, rc_checks.ReleaseError) as exc:
        say_failure(f"REFUSED: {exc}", "refused before running · no score")
        return 1
    if not prep.outcome.ok:
        say("The shared checks reject this submission (the backend would reject it too); fix the findings above first.")
        say("VERDICT: rejected by the shared checks · no score")
        return 1
    limits = PolicyLimits()
    host = (host_factory or LocalDocker)(args, deps, limits)
    try:
        image = launcher.image_ref_from_release(prep.release, host.platform,
                                                source=getattr(prep, "release_source", "release.json"))
    except (launcher.LaunchError, ValueError) as exc:
        say_failure(f"cannot choose the policy image: {exc}", "setup failed · no score")
        return 2
    hf_repo, hf_sha = (prep.hf.repo_id, prep.hf.sha) if prep.hf else (None, None)
    weights_dir = str(prep.weights) if args.weights_dir else None
    prov = {"git_sha": prep.git.sha, "hf_repo": hf_repo, "hf_sha": hf_sha, "weights_dir": weights_dir,
            "image_platform_digest": image.platform_digest, "image_manifest_digest": image.manifest_digest,
            "tree_sha256": prep.tree_sha256}
    engine = EngineClient(args.engine)
    policy: Any = None
    server_started: Optional[float] = None
    try:
        policy, alive, where = host.start(prep, image, prov)
        server_started = time.monotonic()
        say(provenance.provenance_block(git_sha=prep.git.sha, hf_repo=hf_repo, hf_sha=hf_sha, weights_dir=weights_dir,
                                        image_platform_digest=prov["image_platform_digest"], where=where,
                                        manifest_digest=image.manifest_digest, tree_sha256=prep.tree_sha256))
        label = provenance.PRACTICE_LABEL.format(where=where)

        def resource_summary(results: Sequence = ()) -> None:
            for line in resources.summary(getattr(policy, "calls", []), server_started=server_started,
                                          memory_line=host.memory_line(), memory_limit=host.memory_limit,
                                          budget_ended=sum(1 for r in results
                                                           if getattr(r, "end_reason", None) == "act_budget_exceeded"),
                                          transport_note=host.transport_note):
                say(line)
        try:
            outcome = _episodes(args, engine, policy, alive, server_started)
            score, failure = outcome[0], outcome[1]
            results = outcome[2] if len(outcome) > 2 else ()
            startup = outcome[3] if len(outcome) > 3 else None
        except rp.StartupFailure as exc:
            say(f"startup failed ({exc.code}): {exc.message}")
            say(host.startup_log())
            resource_summary()
            say(f"VERDICT: startup failed ({exc.code}) · no score · the policy log above shows why")
            print(_score_json(rp.report({"scorer": None, "total": None, "breakdown": {}}, []), prov, label, str(exc)))
            return 2
        except EngineInfraError as exc:
            say_failure(f"engine error (not your policy's fault): {exc}", "engine error · no score", args.engine)
            return 2
        resource_summary(results)
        say(label)
        print(_score_json(rp.report(score, list(results), startup), prov, label, failure))
        if failure:
            say(f"no contest score: {failure}")
        say(rp.verdict_line(score, list(results), failure))
        return 2 if failure else 0
    except (launcher.LaunchError, DockerError, SystemExit, *host.errors) as exc:
        say_failure(f"practice could not run: {exc}", "practice could not run · no score", args.engine)
        return 2
    finally:
        try:
            host.close()  # first: a host that cannot be cleaned up must not be hidden by another cleanup error
        finally:
            engine.close()
            if policy is not None:
                policy.close()
            if not args.work_dir:
                shutil.rmtree(prep.workdir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="practice", description="Practice launcher for the pinned policy runtime.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="check, launch the published image and run public-task episodes")
    r.add_argument("--repo", default=".", help="your policy repository (a git checkout)")
    r.add_argument("--model", default=None, help="Hugging Face model link (https://huggingface.co/org/name); "
                   "optional with --weights-dir, which then runs local weights and records no model id")
    r.add_argument("--revision", default=None, help="HF revision (exact SHA recommended; default branch if omitted)")
    r.add_argument("--weights-dir", default=None, help="use these local weights (a downloaded snapshot or a local "
                   "checkpoint) instead of downloading")
    r.add_argument("--release", default=None, help="published release manifest (default: $RC_RELEASE_JSON, else "
                   "runtime/release.json shipped in the participant bundle)")
    r.add_argument("--allow-dirty", action="store_true", help="practice the last commit even though the working tree is dirty")
    r.add_argument("--work-dir", default=None, help="keep exported tree and snapshot here (default: temporary)")
    r.add_argument("--engine", default="http://127.0.0.1:8090", help="running public engine (see contestants/PRACTICE.md)")
    r.add_argument("--tasks", default="", help="comma-separated public task ids (default: all the engine serves)")
    r.add_argument("--seeds", default="0-3")
    r.add_argument("--journal-dir", default="")
    r.add_argument("--judged", action="store_true", help="judged-style episode settings (90 s /act budget)")
    r.add_argument("--scenario-seconds", type=float, default=90.0,
                   help="with --judged: per-repetition /act budget in seconds (not the 60 s simulated length)")
    r.add_argument("--max-steps", type=int, default=None,
                   help="end each repetition after this many control steps (a short CPU run; judged runs go up to 1,200)")
    r.add_argument("--warmup-acts", type=int, default=3)
    r.add_argument("--act-timeout", type=float, default=10.0)
    r.add_argument("--ready-timeout", type=float, default=60.0, help="engine readiness wait")
    r.add_argument("--policy-ready-timeout", type=float, default=1800.0, help="policy load/readiness wait")
    r.add_argument("--gpus", default=None, help="docker GPU spec: all (default from runner config), none, or a device list")
    r.add_argument("--memory", default=None, help="policy memory limit (default: the runner's published limit)")
    r.add_argument("--no-pull", action="store_true", help="never pull the image")
    r.add_argument("--print-argv", action="store_true", help="print the exact docker run arguments")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
