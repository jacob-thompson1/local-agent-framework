"""Diff-based file editing tuned for small models.

Small local models cannot reliably emit unified-diff syntax (hunk headers
require line counting). They *can* reproduce a block of text they just read.
So edits here are **search/replace blocks**: the model proposes the exact text
to find and its replacement; the framework does the matching.

Two-tool protocol (propose is cheap and safe, apply is gated):

1. ``propose_edit(path, search, replace)`` -- READ_ONLY. Validates the match
   *now*, stores the pending edit server-side, and returns a short
   ``edit_id`` plus a diff preview. Nothing is written.
2. ``apply_edit(edit_id)`` -- WRITE, gated by your PermissionPolicy. Applies
   the stored edit. Because the model references the edit by id, it never
   pays the diff's tokens twice.

Matching is deliberately forgiving (small models mangle whitespace): exact
match first, then a whitespace-normalized line match. Ambiguous (multiple
occurrences) or missing matches fail with a corrective observation that
includes the closest real text -- the same feed-the-error-back pattern the
JSON loop uses, so the model can self-correct.

Standard library only; no network.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import tool
from .safety import Severity

logger = logging.getLogger("my_agent_framework.editing")

_WS = re.compile(r"[ \t]+")


def _norm_line(line: str) -> str:
    return _WS.sub(" ", line.strip())


@dataclass
class ProposedEdit:
    edit_id: str
    path: Path
    search: str
    replace: str
    created_ts: float = field(default_factory=time.time)
    applied: bool = False

    def diff(self, context: int = 2) -> str:
        """Unified-diff preview of just this block (for humans, not models)."""
        lines = difflib.unified_diff(
            self.search.splitlines(), self.replace.splitlines(),
            fromfile=str(self.path), tofile=str(self.path), lineterm="",
            n=context,
        )
        return "\n".join(lines)


class MatchResult:
    __slots__ = ("start", "end", "kind")

    def __init__(self, start: int, end: int, kind: str) -> None:
        self.start, self.end, self.kind = start, end, kind


def flexible_find(haystack: str, needle: str) -> MatchResult:
    """Locate *needle* in *haystack*: exact, else whitespace-normalized.

    Raises ``ValueError`` with a corrective, model-readable message on zero
    or multiple matches. The multiple-match error asks for more surrounding
    context (the same fix a human would apply).
    """
    # exact
    count = haystack.count(needle)
    if count == 1:
        start = haystack.index(needle)
        return MatchResult(start, start + len(needle), "exact")
    if count > 1:
        raise ValueError(
            f"search text matches {count} locations; include more surrounding "
            "lines so the match is unique."
        )
    # whitespace-normalized, line-wise
    hay_lines = haystack.splitlines(keepends=True)
    norm_hay = [_norm_line(ln) for ln in hay_lines]
    norm_needle = [_norm_line(ln) for ln in needle.splitlines() if _norm_line(ln)]
    if norm_needle:
        hits = []
        window = len(norm_needle)
        compact_hay = [(i, ln) for i, ln in enumerate(norm_hay) if ln]
        compact_lines = [ln for _, ln in compact_hay]
        for j in range(len(compact_lines) - window + 1):
            if compact_lines[j:j + window] == norm_needle:
                hits.append(j)
        if len(hits) == 1:
            first_orig = compact_hay[hits[0]][0]
            last_orig = compact_hay[hits[0] + window - 1][0]
            start = sum(len(ln) for ln in hay_lines[:first_orig])
            end = sum(len(ln) for ln in hay_lines[:last_orig + 1])
            return MatchResult(start, end, "whitespace_normalized")
        if len(hits) > 1:
            raise ValueError(
                f"search text matches {len(hits)} locations (ignoring "
                "whitespace); include more surrounding lines so the match is "
                "unique."
            )
    # nothing -- offer the closest real text so the model can retry
    closest = difflib.get_close_matches(
        _norm_line(needle.splitlines()[0]) if needle.splitlines() else "",
        [ln for ln in norm_hay if ln], n=1, cutoff=0.4,
    )
    hint = f" Closest line in the file: '{closest[0]}'." if closest else ""
    raise ValueError(
        "search text not found in the file, even ignoring whitespace. "
        f"Re-read the file and copy the text exactly.{hint}"
    )


class EditSession:
    """Holds pending edits and manufactures the two bound tools.

    ::

        edits = EditSession(workspace_root="/path/to/project")
        agent = SmallModelAgent(..., tools=[read_file, *edits.tools()])

    ``on_propose`` (if set) is called with ``{edit_id, path, diff}`` after a
    successful proposal -- the agent wires this to the event bus as
    ``diff_proposed`` so UIs can render the diff before approval.
    """

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        max_pending: int = 20,
    ) -> None:
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root else None
        )
        self.max_pending = max_pending
        self.pending: dict[str, ProposedEdit] = {}
        self.on_propose = None  # Optional[Callable[[dict], None]]

    # -- internals ---------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if self.workspace_root is not None and not p.is_relative_to(
            self.workspace_root
        ):
            raise ValueError(
                f"'{p}' is outside the workspace root '{self.workspace_root}'."
            )
        return p

    def describe(self, edit_id: str) -> str:
        edit = self.pending.get(edit_id)
        return edit.diff() if edit else f"(no pending edit '{edit_id}')"

    # -- the tools ---------------------------------------------------------

    def tools(self) -> list:
        session = self

        @tool(
            description=(
                "Propose a file edit. 'search' must be text copied exactly "
                "from the file; 'replace' is the new text. Returns an edit_id."
            ),
            severity=Severity.READ_ONLY,
            keywords=["edit", "modify", "change", "replace", "fix", "file"],
            name="propose_edit",
        )
        def propose_edit(path: str, search: str, replace: str) -> str:
            target = session._resolve(path)
            if not target.is_file():
                return f"Error: '{path}' does not exist or is not a file."
            content = target.read_text(encoding="utf-8", errors="replace")
            try:
                match = flexible_find(content, search)
            except ValueError as exc:
                return f"Error: {exc}"
            if len(session.pending) >= session.max_pending:
                oldest = min(session.pending.values(), key=lambda e: e.created_ts)
                session.pending.pop(oldest.edit_id, None)
            edit = ProposedEdit(
                edit_id=uuid.uuid4().hex[:8], path=target,
                search=content[match.start:match.end], replace=replace,
            )
            session.pending[edit.edit_id] = edit
            if session.on_propose is not None:
                try:
                    session.on_propose({
                        "edit_id": edit.edit_id, "path": str(target),
                        "diff": edit.diff(),
                    })
                except Exception:
                    logger.exception("on_propose callback raised")
            return (
                f"Edit {edit.edit_id} ready for '{target.name}' "
                f"(match: {match.kind}). Call apply_edit with edit_id "
                f"'{edit.edit_id}' to write it."
            )

        @tool(
            description="Apply a previously proposed edit by its edit_id.",
            severity=Severity.WRITE,
            keywords=["edit", "apply", "write", "save", "file"],
            name="apply_edit",
        )
        def apply_edit(edit_id: str) -> str:
            edit = session.pending.get(str(edit_id))
            if edit is None:
                known = sorted(session.pending)
                return (
                    f"Error: no pending edit '{edit_id}'. "
                    f"Pending: {known or 'none'}. Use propose_edit first."
                )
            if edit.applied:
                return f"Error: edit {edit.edit_id} was already applied."
            content = edit.path.read_text(encoding="utf-8", errors="replace")
            try:
                match = flexible_find(content, edit.search)
            except ValueError as exc:
                # File changed between propose and apply.
                session.pending.pop(edit.edit_id, None)
                return (
                    f"Error: file changed since the edit was proposed ({exc}) "
                    "-- propose the edit again."
                )
            new_content = content[:match.start] + edit.replace + content[match.end:]
            edit.path.write_text(new_content, encoding="utf-8", newline="\n")
            edit.applied = True
            session.pending.pop(edit.edit_id, None)
            delta = len(edit.replace.splitlines()) - len(edit.search.splitlines())
            return (
                f"Applied edit {edit.edit_id} to '{edit.path.name}' "
                f"({len(edit.search.splitlines())} line(s) replaced, "
                f"net {delta:+d} line(s))."
            )

        return [propose_edit, apply_edit]
