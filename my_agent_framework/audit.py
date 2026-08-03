"""Compliance-grade audit logging.

Every session writes an append-only JSONL file. Events cover the full
checklist a regulator would ask for:

* **what system** -- model spec, parameters, framework version, active tools
* **when** -- UTC ISO-8601 timestamps on every event
* **who** -- user/role passed at session start
* **what it decided & why** -- every LLM call's reasoning ("thought"), every
  tool selection, every final answer
* **overrides** -- approvals/rejections from the permission layer
* **failures** -- errors, retries, fallbacks (including local->cloud fallback)

Files live under the platform data dir (``platformdirs``):
``.../my-agent-framework/audit/<YYYY-MM-DD>/<session_id>.jsonl``

Exports (JSON/CSV), date-range bulk export, summary statistics, and regex
redaction are provided for regulatory sharing. Nothing here touches the
network.

Retention: the framework never deletes audit files. Insurance audit trails are
commonly retained 3-7 years; see docs/COMPLIANCE_GUIDE.md.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("my_agent_framework.audit")

# Built-in redaction patterns (extend via AuditExporter(redact_patterns=...)).
DEFAULT_REDACTION_PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "phone_us": r"\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_audit_root() -> Path:
    from platformdirs import user_data_dir  # lazy; tiny, offline

    return Path(user_data_dir("my-agent-framework", appauthor=False)) / "audit"


@dataclass
class AuditLogger:
    """Append-only per-session audit writer.

    Create one per agent session. ``event()`` flushes each record to disk
    immediately so a crash never loses accepted events.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user: str = "unknown"
    role: str = "unknown"
    root: Path = field(default_factory=default_audit_root)
    enabled: bool = True
    bus: Optional[Any] = None   # EventBus; every record is mirrored onto it
    _path: Optional[Path] = field(default=None, repr=False)
    _seq: int = field(default=0, repr=False)

    @property
    def path(self) -> Path:
        if self._path is None:
            day = date.today().isoformat()
            directory = self.root / day
            directory.mkdir(parents=True, exist_ok=True)
            self._path = directory / f"{self.session_id}.jsonl"
        return self._path

    def event(self, event_type: str, **payload: Any) -> None:
        """Write one audit record and mirror it onto the event bus (if any).

        This is the framework's single emission point: whatever the bus's
        subscribers (TUI, dashboards, tests) see is exactly what the
        compliance log records. Never raises into the agent loop.
        """
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": _utcnow(),
            "session_id": self.session_id,
            "user": self.user,
            "role": self.role,
            "event": event_type,
            **payload,
        }
        if self.bus is not None:
            self.bus.emit(event_type, record)
        if not self.enabled:
            return
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("Audit write failed for event %r", event_type)

    # -- convenience wrappers used by the agent ---------------------------

    def session_start(
        self, model: str, model_params: dict, tools: list[dict],
        framework_version: str, task: str, config: dict,
    ) -> None:
        self.event(
            "session_start", model=model, model_params=model_params,
            tools_available=tools, framework_version=framework_version,
            task=task, config=config,
        )

    def llm_call(self, iteration: int, prompt_tokens: int, raw_output: str) -> None:
        self.event(
            "llm_call", iteration=iteration,
            prompt_tokens_estimate=prompt_tokens, raw_output=raw_output,
        )

    def decision(self, iteration: int, thought: str, tool: Optional[str],
                 args: Optional[dict], sensitive: bool = False) -> None:
        """The model's reasoning + chosen action. Set ``sensitive=True`` to flag
        decisions that could affect protected classes (bias/fairness review)."""
        self.event(
            "decision", iteration=iteration, thought=thought,
            tool=tool, args=args, sensitive_decision=sensitive,
        )

    def approval(self, tool: str, severity: str, outcome: str,
                 required_approval: bool, detail: str) -> None:
        self.event(
            "approval", tool=tool, severity=severity, outcome=outcome,
            required_approval=required_approval, detail=detail,
        )

    def tool_result(self, tool: str, ok: bool, result: Any,
                    duration_s: float, error: Optional[str] = None) -> None:
        self.event(
            "tool_result", tool=tool, ok=ok,
            result=_truncate(result), duration_s=round(duration_s, 3), error=error,
        )

    def error(self, where: str, error: str, recovery: str = "") -> None:
        self.event("error", where=where, error=error, recovery=recovery)

    def fallback(self, from_model: str, to_model: str, reason: str) -> None:
        self.event("model_fallback", from_model=from_model,
                   to_model=to_model, reason=reason)

    def session_end(self, status: str, final_answer: Optional[str],
                    iterations: int, total_prompt_tokens: int) -> None:
        self.event(
            "session_end", status=status, final_answer=_truncate(final_answer),
            iterations=iterations, total_prompt_tokens_estimate=total_prompt_tokens,
        )


def _truncate(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...[truncated {len(value) - limit} chars]"
    return value


# ---------------------------------------------------------------------------
# Export / reporting
# ---------------------------------------------------------------------------

class AuditExporter:
    """Read, filter, redact, and export audit logs for compliance review."""

    def __init__(self, root: Optional[Path] = None,
                 redact_patterns: Optional[dict[str, str]] = None) -> None:
        self.root = root or default_audit_root()
        self.redact_patterns = redact_patterns

    # -- discovery --------------------------------------------------------

    def iter_records(
        self,
        session_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> Iterator[dict]:
        if not self.root.exists():
            return
        for day_dir in sorted(self.root.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day = date.fromisoformat(day_dir.name)
            except ValueError:
                continue
            if date_from and day < date_from:
                continue
            if date_to and day > date_to:
                continue
            for path in sorted(day_dir.glob("*.jsonl")):
                if session_id and path.stem != session_id:
                    continue
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                logger.warning("Skipping corrupt line in %s", path)

    # -- redaction --------------------------------------------------------

    def _redact_text(self, text: str) -> str:
        assert self.redact_patterns is not None
        for label, pattern in self.redact_patterns.items():
            text = re.sub(pattern, f"[REDACTED:{label}]", text)
        return text

    def _redact(self, obj: Any) -> Any:
        if self.redact_patterns is None:
            return obj
        if isinstance(obj, str):
            return self._redact_text(obj)
        if isinstance(obj, dict):
            return {k: self._redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._redact(v) for v in obj]
        return obj

    # -- exports ----------------------------------------------------------

    def export(
        self,
        fmt: str = "json",
        session_id: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        out_path: Optional[Path] = None,
    ) -> str:
        """Export matching records as a JSON array or CSV string.

        If *out_path* is given the export is also written there (UTF-8).
        """
        records = [self._redact(r) for r in
                   self.iter_records(session_id, date_from, date_to)]
        if fmt == "json":
            text = json.dumps(records, indent=2, ensure_ascii=False, default=str)
        elif fmt == "csv":
            buf = io.StringIO()
            fieldnames = ["seq", "ts", "session_id", "user", "role", "event", "detail"]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                base = {k: r.get(k, "") for k in fieldnames[:-1]}
                extra = {k: v for k, v in r.items() if k not in fieldnames}
                base["detail"] = json.dumps(extra, ensure_ascii=False, default=str)
                writer.writerow(base)
            text = buf.getvalue()
        else:
            raise ValueError(f"Unknown export format {fmt!r} (use 'json' or 'csv')")
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        return text

    # -- summary ----------------------------------------------------------

    def summary(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> dict:
        """Aggregate statistics: sessions, decisions, tool usage, approval and
        error rates, sensitive-decision count."""
        sessions: set[str] = set()
        decisions = 0
        sensitive = 0
        tool_usage: dict[str, int] = {}
        approvals = {"approved": 0, "rejected": 0, "allowed": 0,
                     "blocked": 0, "dry_run": 0}
        errors = 0
        fallbacks = 0
        for r in self.iter_records(None, date_from, date_to):
            sessions.add(r.get("session_id", "?"))
            ev = r.get("event")
            if ev == "decision":
                decisions += 1
                if r.get("sensitive_decision"):
                    sensitive += 1
                if r.get("tool"):
                    tool_usage[r["tool"]] = tool_usage.get(r["tool"], 0) + 1
            elif ev == "approval":
                outcome = r.get("outcome", "")
                if outcome in approvals:
                    approvals[outcome] += 1
            elif ev == "error":
                errors += 1
            elif ev == "model_fallback":
                fallbacks += 1
        asked = approvals["approved"] + approvals["rejected"]
        return {
            "sessions": len(sessions),
            "decisions": decisions,
            "sensitive_decisions": sensitive,
            "tool_usage": dict(sorted(tool_usage.items(), key=lambda kv: -kv[1])),
            "approvals": approvals,
            "approval_rate": round(approvals["approved"] / asked, 3) if asked else None,
            "errors": errors,
            "model_fallbacks": fallbacks,
        }
