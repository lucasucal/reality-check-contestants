"""Exception types shared by the worker, the episode loop and the docker driver."""
from __future__ import annotations


class WorkAborted(Exception):
    """Shutdown requested mid-run: abort local work without writing state. The
    lease is left to expire; the backend sweep owns recovery (plan §3.4)."""


class SubmissionFailed(Exception):
    """The submission fails terminally with `code` (plan §5); reported as
    state=failed with `message` as the progress note."""

    node_fault = False

    def __init__(self, code: str, message: str, team_detail: str | None = None, *,
                 reason: str | None = None, stage: str | None = None, node_fault: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        # Team-safe one-liner (no logs, no hidden-task information) reported as failure_detail.
        # Dropped by the worker when the failure happened in the hidden phase.
        self.team_detail = team_detail
        # Refinement of `code` from FAILURE_REASONS (error-surfacing contract); dropped in the hidden phase.
        self.reason = reason
        # Explicit FAILURE_STAGES value; None = the worker's current stage.
        self.stage = stage
        if node_fault:
            self.node_fault = True


# Closed enums of the error-surfacing contract (docs/verification/error-surfacing-contract.md).
FAILURE_STAGES = frozenset({"prepare", "queue", "start", "warmup", "public_evaluation", "hidden_evaluation",
                            "scoring", "recording", "control"})
READYZ_ERROR_CODES = frozenset({"entry_import_failed", "entry_not_policy", "entry_factory_failed",
                                "reserved_module_shadow", "sdk_shadowed", "load_failed"})
FAILURE_REASONS = frozenset({
    "exit_entry_point_invalid", "exit_reserved_shadow", "program_exited", "program_raised", "never_listened",
    "loopback", "runtime_internal", "load_deadline", "contract_mismatch", "act_http_error", "act_timeout",
    "not_json", "echo_mismatch", "no_action", "too_large", "action_shape", "memory_exceeded", "oom_killed",
    "total_deadline", "lease_lost",
} | {f"readyz_load_failed:{code}" for code in READYZ_ERROR_CODES})


class EngineInfraError(Exception):
    """The engine misbehaved (crash, 5xx, not_ready, step_mismatch, ...). Never the
    team's fault (plan §5 F8/F9): the episode is discarded and retried once."""


class ActionRejected(Exception):
    """The engine refused the policy's action (400 policy_protocol_violation)."""


class ArtifactError(Exception):
    """Context/weights download or extraction failed (infra) or was refused as unsafe."""
