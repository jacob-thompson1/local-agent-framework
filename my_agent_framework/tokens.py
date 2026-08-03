"""Token counting, budgeting, and real-time usage tracking.

Small models live and die by context budget. This module provides:

* :class:`TokenCounter` -- estimates token counts. Uses ``tiktoken`` if it is
  installed (accurate for OpenAI-family tokenizers, a good proxy for most
  others), otherwise falls back to a chars/3.6 heuristic that is deliberately
  slightly pessimistic so budgets fail safe.
* :class:`ContextBudget` -- tracks how much of the model's context window is
  consumed by system prompt, tool definitions, history, and the current turn,
  and emits warnings as thresholds are crossed.

No network calls are made anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("my_agent_framework.tokens")

# Pessimistic heuristic: English prose averages ~4 chars/token; code and JSON
# average lower. 3.6 keeps estimates on the safe (over-counting) side.
_CHARS_PER_TOKEN = 3.6


class TokenCounter:
    """Estimate token counts without any network access.

    Attempts to lazily load ``tiktoken`` on first use. If unavailable (it is an
    optional dependency), falls back to a character-based heuristic.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoder = None
        self._tried_tiktoken = False

    def _get_encoder(self):
        if not self._tried_tiktoken:
            self._tried_tiktoken = True
            try:
                import tiktoken  # noqa: PLC0415 -- intentional lazy import

                self._encoder = tiktoken.get_encoding(self._encoding_name)
                logger.debug("tiktoken loaded (%s)", self._encoding_name)
            except Exception:  # ImportError or missing encoding data
                logger.debug(
                    "tiktoken unavailable; using chars/%.1f heuristic",
                    _CHARS_PER_TOKEN,
                )
        return self._encoder

    def count(self, text: str) -> int:
        """Return an estimated token count for *text*."""
        if not text:
            return 0
        encoder = self._get_encoder()
        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception:  # pragma: no cover - defensive
                pass
        return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass
class BudgetReport:
    """Snapshot of context consumption at a point in time."""

    context_window: int
    system_tokens: int
    tool_tokens: int
    history_tokens: int
    reserved_for_output: int

    @property
    def used(self) -> int:
        return self.system_tokens + self.tool_tokens + self.history_tokens

    @property
    def available(self) -> int:
        return self.context_window - self.used - self.reserved_for_output

    @property
    def utilization(self) -> float:
        """Fraction of the *input* budget consumed (0.0-1.0+)."""
        input_budget = self.context_window - self.reserved_for_output
        if input_budget <= 0:
            return 1.0
        return self.used / input_budget

    def as_dict(self) -> dict:
        return {
            "context_window": self.context_window,
            "system_tokens": self.system_tokens,
            "tool_tokens": self.tool_tokens,
            "history_tokens": self.history_tokens,
            "reserved_for_output": self.reserved_for_output,
            "used": self.used,
            "available": self.available,
            "utilization": round(self.utilization, 3),
        }


@dataclass
class ContextBudget:
    """Tracks context-window consumption and warns as limits approach.

    Parameters
    ----------
    context_window:
        Total context length of the model in tokens (e.g. 4096, 8192, 32768).
    reserved_for_output:
        Tokens held back for the model's response each turn.
    warn_at / critical_at:
        Utilization thresholds (fractions) at which warnings are logged.
    """

    context_window: int = 4096
    reserved_for_output: int = 512
    warn_at: float = 0.75
    critical_at: float = 0.90
    counter: TokenCounter = field(default_factory=TokenCounter)
    _warned: set = field(default_factory=set, repr=False)

    def report(
        self,
        system_text: str = "",
        tool_text: str = "",
        history_texts: Optional[list[str]] = None,
    ) -> BudgetReport:
        """Compute a :class:`BudgetReport` and log threshold warnings once each."""
        report = BudgetReport(
            context_window=self.context_window,
            system_tokens=self.counter.count(system_text),
            tool_tokens=self.counter.count(tool_text),
            history_tokens=sum(self.counter.count(t) for t in (history_texts or [])),
            reserved_for_output=self.reserved_for_output,
        )
        util = report.utilization
        if util >= self.critical_at and "critical" not in self._warned:
            self._warned.add("critical")
            logger.warning(
                "Context utilization CRITICAL: %.0f%% of input budget used "
                "(%d/%d tokens). History will be trimmed aggressively; consider "
                "fewer tools or a shorter task description.",
                util * 100, report.used, self.context_window - self.reserved_for_output,
            )
        elif util >= self.warn_at and "warn" not in self._warned:
            self._warned.add("warn")
            logger.warning(
                "Context utilization high: %.0f%% of input budget used "
                "(%d tokens). Small models degrade well before 100%%.",
                util * 100, report.used,
            )
        return report

    def reset_warnings(self) -> None:
        self._warned.clear()


# ---------------------------------------------------------------------------
# Pricing (for cost visibility -- see models.CostTracker)
# ---------------------------------------------------------------------------

#: Prices go stale. These defaults exist so cost tracking works out of the box,
#: but they are snapshots -- verify against your provider's current price list
#: and override via ``CostTracker(pricing=...)`` or the ``pricing`` setting
#: (``my-agent config set pricing '{"gpt-4o-mini": [0.15, 0.6]}'``). The
#: framework never fetches prices from the network.
PRICING_AS_OF = "2026-07 (verify before relying on these numbers)"

#: model-name substring -> (USD per 1M input tokens, USD per 1M output tokens).
#: Matched longest-substring-first against the lowercased model name. Local
#: providers (ollama, custom) are always $0 regardless of this table.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "haiku": (0.80, 4.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
}


def lookup_price(
    model_name: str,
    pricing: Optional[dict] = None,
) -> Optional[tuple[float, float]]:
    """Return ``(input_per_1m, output_per_1m)`` for *model_name*, or None.

    Longest matching substring wins, so ``"gpt-4o-mini"`` beats ``"gpt-4o"``.
    """
    table = pricing if pricing is not None else DEFAULT_PRICING
    name = model_name.lower()
    best: Optional[tuple[float, float]] = None
    best_len = -1
    for key, value in table.items():
        if key.lower() in name and len(key) > best_len:
            best = (float(value[0]), float(value[1]))
            best_len = len(key)
    return best


def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Optional[dict] = None,
) -> Optional[float]:
    """Estimated USD cost of one call, or None if the model isn't priced."""
    price = lookup_price(model_name, pricing)
    if price is None:
        return None
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
