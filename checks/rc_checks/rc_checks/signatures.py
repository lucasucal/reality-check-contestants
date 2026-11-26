"""Known-issue failure signatures (docs/runtime-interfaces.md section 7).

Matching rule: signatures are tried in file order and the FIRST entry with any matching
log pattern (or whose exit_codes contains the exit code) wins. Put more specific entries
before generic ones. Matching is deterministic and stdlib only.

Cost bounds: only the last MAX_LOG_CHARS of the log are scanned; at most MAX_SIGNATURES
entries; regex patterns are limited in length and must not use unbounded `.*` / `.+` or
quantified groups (the usual catastrophic-backtracking shapes). Substrings are matched
case-sensitively; regexes are matched with `re.search`.

Team-visible output is only the issue id and a link; the signature name and the log never
leave the organizer side.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_SIGNATURES = 40
MAX_LOG_CHARS = 20000
MAX_PATTERN_CHARS = 200
_ID_RE = re.compile(r"^KSI-\d{3,}$")
_UNBOUNDED = re.compile(r"(?<!\\)\.[*+]|(?<!\\)\)[*+{]")


class SignatureError(Exception):
    pass


@dataclass(frozen=True)
class KnownIssueMatch:
    issue_id: str
    signature: str  # organizer-only


def default_path() -> Path:
    env = os.environ.get("RC_FAILURE_SIGNATURES")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "known-issues" / "signatures.json"


def _validate(data: object) -> list:
    if not isinstance(data, dict) or not isinstance(data.get("signatures"), list):
        raise SignatureError("signatures file must be an object with a 'signatures' list")
    sigs = data["signatures"]
    if len(sigs) > MAX_SIGNATURES:
        raise SignatureError(f"too many signatures ({len(sigs)} > {MAX_SIGNATURES})")
    for i, sig in enumerate(sigs):
        if not isinstance(sig, dict):
            raise SignatureError(f"signature {i} is not an object")
        if not isinstance(sig.get("issue_id"), str) or not _ID_RE.match(sig["issue_id"]):
            raise SignatureError(f"signature {i}: bad issue_id")
        if not isinstance(sig.get("name"), str) or not sig["name"]:
            raise SignatureError(f"signature {i}: missing name")
        codes = sig.get("exit_codes", [])
        if not isinstance(codes, list) or not all(isinstance(c, int) and not isinstance(c, bool) for c in codes):
            raise SignatureError(f"signature {i}: exit_codes must be a list of ints")
        pats = sig.get("log_patterns", [])
        if not isinstance(pats, list):
            raise SignatureError(f"signature {i}: log_patterns must be a list")
        for p in pats:
            if not isinstance(p, dict):
                raise SignatureError(f"signature {i}: bad pattern")
            kind = p.get("type")
            if kind == "substring":
                text = p.get("text")
                if not isinstance(text, str) or not text or len(text) > MAX_PATTERN_CHARS:
                    raise SignatureError(f"signature {i}: bad substring")
            elif kind == "regex":
                pat = p.get("pattern")
                if not isinstance(pat, str) or not pat or len(pat) > MAX_PATTERN_CHARS:
                    raise SignatureError(f"signature {i}: bad regex")
                if _UNBOUNDED.search(pat):
                    raise SignatureError(f"signature {i}: regex must not use unbounded .* / .+ or quantified groups")
                try:
                    re.compile(pat)
                except re.error as exc:
                    raise SignatureError(f"signature {i}: invalid regex: {exc}") from exc
            else:
                raise SignatureError(f"signature {i}: unknown pattern type {kind!r}")
    return sigs


def load_signatures(path: Optional[Path] = None) -> list:
    p = Path(path) if path is not None else default_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SignatureError(f"cannot read signatures {p}: {exc}") from exc
    return _validate(data)


def _matches(sig: dict, log: str, exit_code: Optional[int]) -> bool:
    if exit_code is not None and exit_code in sig.get("exit_codes", []):
        return True
    for p in sig.get("log_patterns", []):
        if p["type"] == "substring":
            if p["text"] in log:
                return True
        elif re.search(p["pattern"], log):
            return True
    return False


def match_failure(log_tail: str, exit_code: Optional[int], *,
                  signatures: Optional[list] = None) -> Optional[KnownIssueMatch]:
    sigs = _validate({"signatures": signatures}) if signatures is not None else load_signatures()
    log = (log_tail or "")[-MAX_LOG_CHARS:]
    for sig in sigs:
        if _matches(sig, log, exit_code):
            return KnownIssueMatch(issue_id=sig["issue_id"], signature=sig["name"])
    return None


def team_text(match: KnownIssueMatch, base_url: str = "/known-issues") -> str:
    return f"This looks like known issue {match.issue_id}: {base_url}#{match.issue_id.lower()}"
