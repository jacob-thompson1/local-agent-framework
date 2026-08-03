"""User control & safety: severity levels, approval policy, dry-run.

Mirrors the approve-before-act pattern: tools are classified READ_ONLY /
WRITE / DESTRUCTIVE, a :class:`PermissionPolicy` decides which classes (or
specific tools) require human confirmation, and a pluggable
``confirmer`` callback collects the yes/no. Every decision -- allowed,
approved, rejected, blocked, dry-run -- is returned as a structured
:class:`PermissionDecision` that the agent writes to the audit log.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("my_agent_framework.safety")


class Severity(enum.Enum):
    """How much damage a tool can do."""

    READ_ONLY = "read_only"      # observes state; no side effects
    WRITE = "write"              # creates/modifies data (files, DB rows, messages)
    DESTRUCTIVE = "destructive"  # deletes data, executes arbitrary code, spends money

    @property
    def rank(self) -> int:
        return {"read_only": 0, "write": 1, "destructive": 2}[self.value]


#: Signature for confirmation callbacks. Receives (tool_name, args, severity,
#: reason_text) and returns True to approve. The default implementation prompts
#: on stdin; GUI/web apps supply their own.
Confirmer = Callable[[str, dict, Severity, str], bool]


def stdin_confirmer(tool_name: str, args: dict, severity: Severity, reason: str) -> bool:
    """Default interactive confirmer: prompts on the terminal."""
    print(f"\n[APPROVAL REQUIRED] {severity.value.upper()} action")
    print(f"  Tool: {tool_name}")
    print(f"  Args: {args}")
    if reason:
        print(f"  Agent's reasoning: {reason}")
    answer = input("  Allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


@dataclass
class PermissionDecision:
    """Outcome of the policy check for one proposed tool call."""

    tool_name: str
    severity: Severity
    outcome: str          # "allowed" | "approved" | "rejected" | "blocked" | "dry_run"
    required_approval: bool
    detail: str = ""

    @property
    def may_execute(self) -> bool:
        return self.outcome in ("allowed", "approved")


@dataclass
class PermissionPolicy:
    """Which actions run freely, which require confirmation, which are banned.

    Parameters
    ----------
    confirm_at_or_above:
        Every tool at this severity or higher requires confirmation.
        ``None`` disables severity-based confirmation.
    always_confirm / never_confirm:
        Per-tool-name overrides (never_confirm wins only below the block level).
    max_severity:
        Hard ceiling -- tools above this severity are *blocked* outright,
        regardless of approval. E.g. ``Severity.READ_ONLY`` yields a
        "read-only agent, no writes allowed".
    dry_run:
        If True, nothing executes; the agent reports what it *would* do.
    confirmer:
        Callback that collects approval. Defaults to a stdin prompt.
    """

    confirm_at_or_above: Optional[Severity] = Severity.WRITE
    always_confirm: set[str] = field(default_factory=set)
    never_confirm: set[str] = field(default_factory=set)
    max_severity: Severity = Severity.DESTRUCTIVE
    dry_run: bool = False
    confirmer: Confirmer = stdin_confirmer

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        severity: Severity,
        reason: str = "",
    ) -> PermissionDecision:
        """Evaluate a proposed tool call. May block, ask the user, or allow."""
        if severity.rank > self.max_severity.rank:
            logger.warning(
                "BLOCKED %s call to %r: exceeds policy ceiling (%s > %s)",
                severity.value, tool_name, severity.value, self.max_severity.value,
            )
            return PermissionDecision(
                tool_name, severity, "blocked", required_approval=False,
                detail=f"severity {severity.value} exceeds policy max "
                       f"{self.max_severity.value}",
            )

        if self.dry_run:
            logger.info("DRY-RUN: would call %s(%s)", tool_name, args)
            return PermissionDecision(
                tool_name, severity, "dry_run", required_approval=False,
                detail="dry_run mode: call simulated, not executed",
            )

        needs_approval = tool_name in self.always_confirm or (
            self.confirm_at_or_above is not None
            and severity.rank >= self.confirm_at_or_above.rank
            and tool_name not in self.never_confirm
        )
        if not needs_approval:
            return PermissionDecision(tool_name, severity, "allowed", False)

        approved = self.confirmer(tool_name, args, severity, reason)
        outcome = "approved" if approved else "rejected"
        logger.info("User %s %s call to %r", outcome, severity.value, tool_name)
        return PermissionDecision(
            tool_name, severity, outcome, required_approval=True,
            detail="user decision via confirmer callback",
        )


def read_only_policy() -> PermissionPolicy:
    """Convenience: agent may only observe. Writes and destructive tools blocked."""
    return PermissionPolicy(max_severity=Severity.READ_ONLY, confirm_at_or_above=None)


def paranoid_policy(confirmer: Optional[Confirmer] = None) -> PermissionPolicy:
    """Convenience: confirm everything, even reads."""
    return PermissionPolicy(
        confirm_at_or_above=Severity.READ_ONLY,
        confirmer=confirmer or stdin_confirmer,
    )
