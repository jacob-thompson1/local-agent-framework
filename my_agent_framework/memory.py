"""Conversation/context memory with token-aware trimming.

Small models cannot afford unbounded history. :class:`ConversationMemory`
keeps a rolling transcript of turns (user, assistant thoughts/actions, tool
observations) and trims oldest-first to fit a token budget. Optionally, a
summarization callback can compress trimmed content into a running summary
instead of dropping it (off by default: an extra LLM call per trim is a real
cost on local hardware).

Everything trimmed is logged so users can see exactly what the model no longer
knows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .tokens import TokenCounter

logger = logging.getLogger("my_agent_framework.memory")

Role = Literal["user", "assistant", "tool"]


@dataclass
class Turn:
    role: Role
    content: str
    pinned: bool = False   # pinned turns (e.g. the original task) never trim


@dataclass
class ConversationMemory:
    """Rolling, token-budgeted transcript.

    Parameters
    ----------
    max_tokens:
        Budget for the rendered history block of the prompt.
    counter:
        Shared :class:`TokenCounter`.
    summarizer:
        Optional ``(text_of_trimmed_turns) -> short_summary`` callable. When
        provided, trimmed content is compressed into ``self.summary`` rather
        than discarded.
    """

    max_tokens: int = 1500
    counter: TokenCounter = field(default_factory=TokenCounter)
    summarizer: Optional[Callable[[str], str]] = None
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    trimmed_turn_count: int = 0
    #: With a summarizer attached, compaction triggers *proactively* at this
    #: fraction of the budget (matching ContextBudget's 75% warning) instead
    #: of waiting for the hard limit.
    warn_ratio: float = 0.75
    #: When compacting, trim down to this fraction (hysteresis: without it,
    #: memory would sit at the threshold and re-compact every single turn).
    target_ratio: float = 0.5
    #: Structured record of every compaction/trim -- what was removed, when,
    #: and the token effect. The agent mirrors entries into the audit log,
    #: because compaction changes what the model saw.
    compaction_history: list[dict] = field(default_factory=list)
    #: Optional ``(entry_dict) -> None`` called after each compaction.
    on_compaction: Optional[Callable[[dict], None]] = None

    def add(self, role: Role, content: str, pinned: bool = False) -> None:
        self.turns.append(Turn(role=role, content=content, pinned=pinned))
        self._enforce_budget()

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        """History block injected into the prompt (plain, delimiter-friendly)."""
        parts: list[str] = []
        if self.summary:
            parts.append(f"[Earlier context, summarized] {self.summary}")
        for turn in self.turns:
            label = {"user": "User", "assistant": "Assistant",
                     "tool": "Observation"}[turn.role]
            parts.append(f"{label}: {turn.content}")
        return "\n".join(parts)

    def token_count(self) -> int:
        return self.counter.count(self.render())

    # -- trimming ----------------------------------------------------------

    def _enforce_budget(self) -> None:
        # Proactive when a summarizer exists (compacting early is cheap and
        # keeps quality); purely reactive otherwise (trimming early would
        # just discard context sooner than necessary).
        proactive = self.summarizer is not None
        trigger = int(self.max_tokens * self.warn_ratio) if proactive \
            else self.max_tokens
        tokens_before = self.token_count()
        if tokens_before <= trigger:
            return
        target = int(self.max_tokens * self.target_ratio) if proactive \
            else self.max_tokens

        trimmed: list[Turn] = []
        # Drop oldest unpinned turns until under target (always keep the most
        # recent 2 turns so the model retains the immediate exchange).
        while self.token_count() > target:
            candidates = [
                i for i, t in enumerate(self.turns[:-2]) if not t.pinned
            ]
            if not candidates:
                if self.token_count() > self.max_tokens:
                    logger.warning(
                        "Memory over budget (%d > %d tokens) but nothing "
                        "trimmable remains (pinned/recent turns only).",
                        self.token_count(), self.max_tokens,
                    )
                break
            trimmed.append(self.turns.pop(candidates[0]))
        if not trimmed:
            return
        self.trimmed_turn_count += len(trimmed)
        trimmed_text = "\n".join(f"{t.role}: {t.content}" for t in trimmed)
        summary_added = ""
        if self.summarizer is not None:
            try:
                summary_added = self.summarizer(trimmed_text)
                self.summary = (self.summary + " " + summary_added).strip()
            except Exception:
                logger.exception("Summarizer failed; trimmed content dropped.")
        entry = {
            "trigger": "proactive_75pct" if proactive else "hard_budget",
            "trimmed_turns": len(trimmed),
            "trimmed_roles": [t.role for t in trimmed],
            "tokens_before": tokens_before,
            "tokens_after": self.token_count(),
            "summary_added": summary_added,
        }
        self.compaction_history.append(entry)
        logger.info(
            "Compacted memory: %d turn(s), %d -> %d tokens (%s).",
            len(trimmed), tokens_before, entry["tokens_after"], entry["trigger"],
        )
        if self.on_compaction is not None:
            try:
                self.on_compaction(entry)
            except Exception:
                logger.exception("on_compaction callback raised; continuing.")

    def truncate_to(self, n_turns: int) -> int:
        """Drop all turns after index *n_turns* (task-only checkpoint restore).

        Returns how many turns were removed. The running summary is left
        intact -- it describes turns *before* any checkpoint.
        """
        removed = max(0, len(self.turns) - n_turns)
        del self.turns[n_turns:]
        if removed:
            logger.info("Truncated memory back to %d turn(s) (-%d).",
                        n_turns, removed)
        return removed

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.trimmed_turn_count = 0
        self.compaction_history.clear()
