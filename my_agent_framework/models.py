"""Model configuration: profiles, family hints, and lazy provider loading.

Design rules honored here:

* **Model-agnostic.** Any ``langchain_core`` ``BaseChatModel`` can be passed
  directly to the agent. Alternatively a provider string like
  ``"ollama:mistral:7b"`` is resolved to a chat model *lazily* at first call.
* **Zero external calls at import/startup.** Provider SDKs (``langchain_ollama``,
  ``langchain_openai``, ``langchain_anthropic``) are imported only inside
  builder functions, and even then no network request occurs until the agent
  actually invokes the model.
* **Hybrid mode.** :class:`HybridChatModel` tries a local model first and falls
  back to a configured cloud model on failure -- never silently: every
  fallback is logged and surfaced in the audit trail.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # import only for type checkers; no runtime cost
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger("my_agent_framework.models")


# ---------------------------------------------------------------------------
# Size profiles: recommended limits by parameter count
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProfile:
    """Recommended operating limits for a model size class."""

    size_class: str                 # "3b", "5b", "7b", "large"
    max_tools: int                  # hard recommendation for tool count
    recommended_tools: int          # comfortable tool count
    context_window: int             # conservative default context assumption
    max_iterations: int             # reasoning-loop iterations before stopping
    few_shot_examples: int          # in-context examples to include (0-2)
    notes: str = ""


PROFILES: dict[str, ModelProfile] = {
    "3b": ModelProfile(
        size_class="3b", max_tools=4, recommended_tools=3, context_window=4096,
        max_iterations=5, few_shot_examples=1,
        notes=(
            "3B models need exactly one job per session. Keep tool descriptions "
            "under ~40 tokens each, always include 1 few-shot example, and "
            "expect to retry malformed JSON once or twice per session."
        ),
    ),
    "5b": ModelProfile(
        size_class="5b", max_tools=6, recommended_tools=4, context_window=8192,
        max_iterations=7, few_shot_examples=1,
        notes=(
            "Mid-size models (4-6B, e.g. Phi-3-mini class) handle 4-6 tools "
            "reliably if descriptions are terse. One few-shot example still "
            "measurably improves tool-call formatting."
        ),
    ),
    "7b": ModelProfile(
        size_class="7b", max_tools=8, recommended_tools=6, context_window=8192,
        max_iterations=10, few_shot_examples=1,
        notes=(
            "7-8B models are the sweet spot for local agents: 6-8 tools, "
            "multi-step plans up to ~10 iterations. Instruction-tuned variants "
            "(Mistral-7B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) "
            "strongly preferred over base models."
        ),
    ),
    "large": ModelProfile(
        size_class="large", max_tools=20, recommended_tools=12,
        context_window=32768, max_iterations=15, few_shot_examples=0,
        notes="13B+ or hosted frontier models; tool pruning is optional.",
    ),
}


# Behavioral hints by model family, injected (briefly) into docs and available
# programmatically. These are guidance, not magic switches.
FAMILY_HINTS: dict[str, str] = {
    "mistral": (
        "Mistral instruct models follow JSON-only instructions well but tend to "
        "add trailing prose after JSON; the parser here strips it. Keep "
        "temperature <=0.3 for tool calling."
    ),
    "llama": (
        "Llama 3.x models occasionally wrap JSON in markdown fences and may "
        "'narrate' before acting; the corrective-retry loop handles this. They "
        "respond well to ALL-CAPS constraint words (ONLY, EXACTLY)."
    ),
    "qwen": (
        "Qwen2.5 models are the strongest small-model JSON emitters and handle "
        "the most tools per size class, but may answer in Chinese if the task "
        "is ambiguous -- state the output language in the task if it matters."
    ),
    "phi": (
        "Phi-3/4 models are verbose reasoners; cap max_tokens per turn and "
        "expect longer 'thought' fields. Strong at code tools, weaker at "
        "multi-step tool chains."
    ),
    "gemma": (
        "Gemma models dislike system-role messages in some serving stacks; if "
        "tool calling fails consistently, merge the system prompt into the "
        "first user message (set merge_system_into_user=True on the agent)."
    ),
}


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def detect_size_class(model_name: str) -> str:
    """Best-effort parameter-size detection from a model name.

    ``"mistral:7b-instruct"`` -> ``"7b"``; unknown names default to ``"7b"``
    (the middle-of-the-road profile) with a logged notice.
    """
    match = _SIZE_RE.search(model_name)
    if match:
        size = float(match.group(1))
        if size <= 3.9:
            return "3b"
        if size <= 6.0:
            return "5b"
        if size <= 9.0:
            return "7b"
        return "large"
    lowered = model_name.lower()
    if any(k in lowered for k in ("gpt-", "claude", "gemini", "o1", "o3")):
        return "large"
    logger.info(
        "Could not infer parameter count from model name %r; assuming the "
        "'7b' profile. Pass size_class explicitly to override.", model_name,
    )
    return "7b"


def detect_family(model_name: str) -> Optional[str]:
    lowered = model_name.lower()
    for family in FAMILY_HINTS:
        if family in lowered:
            return family
    return None


# ---------------------------------------------------------------------------
# Provider string parsing and lazy construction
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Parsed provider string, e.g. ``ollama:mistral:7b`` or ``openai:gpt-4o-mini``."""

    provider: str
    model: str
    raw: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        if ":" not in spec:
            raise ValueError(
                f"Model spec {spec!r} must be 'provider:model', e.g. "
                "'ollama:mistral:7b', 'openai:gpt-4o-mini', 'anthropic:claude-haiku-4-5'."
            )
        provider, model = spec.split(":", 1)
        return cls(provider=provider.lower(), model=model, raw=spec)


def _build_ollama(model: str, **kwargs: Any) -> "BaseChatModel":
    try:
        from langchain_ollama import ChatOllama  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Ollama support requires 'langchain-ollama'. "
            "Install with: pip install my-agent-framework[ollama]"
        ) from exc
    kwargs.setdefault("temperature", 0.2)
    return ChatOllama(model=model, **kwargs)


def _build_openai(model: str, **kwargs: Any) -> "BaseChatModel":
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenAI support requires 'langchain-openai'. "
            "Install with: pip install my-agent-framework[openai]"
        ) from exc
    kwargs.setdefault("temperature", 0.2)
    return ChatOpenAI(model=model, **kwargs)


def _build_anthropic(model: str, **kwargs: Any) -> "BaseChatModel":
    try:
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Anthropic support requires 'langchain-anthropic'. "
            "Install with: pip install my-agent-framework[anthropic]"
        ) from exc
    kwargs.setdefault("temperature", 0.2)
    return ChatAnthropic(model=model, **kwargs)


def _build_openai_compatible(model: str, **kwargs: Any) -> "BaseChatModel":
    """Custom/self-hosted endpoints speaking the OpenAI protocol (vLLM,
    llama.cpp server, LM Studio, TGI). Requires ``base_url`` in kwargs."""
    if "base_url" not in kwargs:
        raise ValueError(
            "Provider 'custom' requires model_kwargs={'base_url': 'http://...'} "
            "pointing at an OpenAI-compatible endpoint."
        )
    kwargs.setdefault("api_key", "not-needed")
    return _build_openai(model, **kwargs)


_BUILDERS: dict[str, Callable[..., "BaseChatModel"]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "custom": _build_openai_compatible,
}

# Providers that run on the local machine (no data leaves the host).
LOCAL_PROVIDERS = frozenset({"ollama", "custom"})


def build_chat_model(spec: str, **model_kwargs: Any) -> "BaseChatModel":
    """Resolve a provider string to a LangChain chat model.

    Importing provider SDKs happens here (lazily), and constructing the client
    object performs **no network I/O** -- requests happen only when the agent
    invokes the model.
    """
    parsed = ModelSpec.parse(spec)
    builder = _BUILDERS.get(parsed.provider)
    if builder is None:
        raise ValueError(
            f"Unknown provider {parsed.provider!r}. Known: {sorted(_BUILDERS)}. "
            "Or pass a BaseChatModel instance directly."
        )
    return builder(parsed.model, **model_kwargs)


# ---------------------------------------------------------------------------
# Hybrid local-first model with cloud fallback
# ---------------------------------------------------------------------------

class HybridChatModel:
    """Local-first model with explicit, logged cloud fallback.

    Not a ``BaseChatModel`` subclass by design -- it exposes the one method the
    agent needs (:meth:`invoke`) plus metadata, keeping the failover logic
    simple and auditable.

    Parameters
    ----------
    primary_spec / fallback_spec:
        Provider strings. The fallback is only *constructed* on first use and
        only *called* if the primary raises.
    on_fallback:
        Optional callback ``(exception) -> None`` invoked before falling back
        (used by the agent to write an audit event).
    """

    def __init__(
        self,
        primary_spec: str,
        fallback_spec: Optional[str] = None,
        primary_kwargs: Optional[dict] = None,
        fallback_kwargs: Optional[dict] = None,
        on_fallback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.primary_spec = primary_spec
        self.fallback_spec = fallback_spec
        self._primary_kwargs = primary_kwargs or {}
        self._fallback_kwargs = fallback_kwargs or {}
        self._primary: Optional["BaseChatModel"] = None
        self._fallback: Optional["BaseChatModel"] = None
        self.on_fallback = on_fallback
        self.last_provider_used: Optional[str] = None
        #: Optional CostTracker. Each call is cap-checked (cloud only) before
        #: dispatch and recorded after, using real usage metadata when the
        #: provider returns it, else TokenCounter estimates.
        self.cost_tracker: Optional["CostTracker"] = None

    @property
    def model_name(self) -> str:
        return self.primary_spec

    def _get_primary(self) -> "BaseChatModel":
        if self._primary is None:
            self._primary = build_chat_model(self.primary_spec, **self._primary_kwargs)
        return self._primary

    def _get_fallback(self) -> "BaseChatModel":
        if self._fallback is None:
            assert self.fallback_spec is not None
            self._fallback = build_chat_model(self.fallback_spec, **self._fallback_kwargs)
        return self._fallback

    def _estimate_prompt_tokens(self, messages: Any) -> int:
        from .tokens import TokenCounter
        counter = TokenCounter()
        if isinstance(messages, str):
            return counter.count(messages)
        try:
            return sum(
                counter.count(str(getattr(m, "content", m))) for m in messages
            )
        except TypeError:
            return counter.count(str(messages))

    def _track(self, spec: str, messages: Any, result: Any) -> None:
        if self.cost_tracker is None:
            return
        usage = _usage_from_response(result)
        if usage is None:
            from .tokens import TokenCounter
            usage = (
                self._estimate_prompt_tokens(messages),
                TokenCounter().count(str(getattr(result, "content", result))),
            )
        self.cost_tracker.record(spec, *usage)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        if self.cost_tracker is not None:
            self.cost_tracker.check_cap(
                self.primary_spec, self._estimate_prompt_tokens(messages)
            )
        try:
            result = self._get_primary().invoke(messages, **kwargs)
            self.last_provider_used = self.primary_spec
            self._track(self.primary_spec, messages, result)
            return result
        except SpendCapExceeded:
            raise
        except Exception as exc:
            if not self.fallback_spec:
                raise
            logger.warning(
                "Primary model %s failed (%s: %s); falling back to %s. "
                "NOTE: data in this request is now leaving the local machine.",
                self.primary_spec, type(exc).__name__, exc, self.fallback_spec,
            )
            if self.on_fallback is not None:
                try:
                    self.on_fallback(exc)
                except Exception:  # pragma: no cover
                    logger.exception("on_fallback callback raised")
            if self.cost_tracker is not None:
                self.cost_tracker.check_cap(
                    self.fallback_spec, self._estimate_prompt_tokens(messages)
                )
            result = self._get_fallback().invoke(messages, **kwargs)
            self.last_provider_used = self.fallback_spec
            self._track(self.fallback_spec, messages, result)
            return result


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class SpendCapExceeded(RuntimeError):
    """Raised before a cloud call that would push spend past the cap."""


@dataclass
class CostTracker:
    """Running cost/token accumulator with an optional hard spend cap.

    Attach to a :class:`HybridChatModel` (``cost_tracker=``) or drive it
    yourself via :meth:`record`. Local providers (``ollama``, ``custom``) are
    tracked as $0.00 but their tokens still count -- free is not the same as
    invisible.

    Token counts are estimates from :class:`~my_agent_framework.tokens.TokenCounter`
    unless the provider response carries real usage metadata (used when
    present). Prices come from ``pricing`` (defaults to
    :data:`~my_agent_framework.tokens.DEFAULT_PRICING`; see
    ``PRICING_AS_OF`` -- verify and override, prices go stale).

    spend_cap_usd:
        Hard ceiling. A *cloud* call that would exceed it raises
        :class:`SpendCapExceeded` **before** the request is made; local calls
        are never blocked. ``on_cap`` (if set) is called first.
    on_cost:
        ``(payload_dict) -> None`` after every recorded call -- the agent
        wires this to the event bus as ``cost_update``.
    """

    pricing: Optional[dict] = None
    spend_cap_usd: Optional[float] = None
    on_cost: Optional[Callable[[dict], None]] = None
    on_cap: Optional[Callable[[dict], None]] = None
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    calls: int = 0
    by_provider: dict = field(default_factory=dict)

    def _price_for(self, spec: str) -> tuple[float, float]:
        from .tokens import lookup_price
        provider = spec.split(":", 1)[0].lower()
        if provider in LOCAL_PROVIDERS:
            return (0.0, 0.0)
        return lookup_price(spec, self.pricing) or (0.0, 0.0)

    def projected_call_cost(
        self, spec: str, prompt_tokens: int, completion_tokens_guess: int = 512
    ) -> float:
        pin, pout = self._price_for(spec)
        return (prompt_tokens * pin + completion_tokens_guess * pout) / 1_000_000

    def check_cap(self, spec: str, prompt_tokens: int) -> None:
        """Raise :class:`SpendCapExceeded` if a call to *spec* would bust the cap."""
        provider = spec.split(":", 1)[0].lower()
        if self.spend_cap_usd is None or provider in LOCAL_PROVIDERS:
            return
        projected = self.total_cost_usd + self.projected_call_cost(spec, prompt_tokens)
        if projected > self.spend_cap_usd:
            detail = {
                "spec": spec, "spend_so_far_usd": round(self.total_cost_usd, 6),
                "cap_usd": self.spend_cap_usd,
                "projected_usd": round(projected, 6),
            }
            if self.on_cap is not None:
                try:
                    self.on_cap(detail)
                except Exception:
                    logger.exception("on_cap callback raised")
            raise SpendCapExceeded(
                f"Spend cap ${self.spend_cap_usd:.2f} would be exceeded by a "
                f"call to {spec} (spent ${self.total_cost_usd:.4f}, projected "
                f"${projected:.4f}). Local models remain available."
            )

    def record(
        self, spec: str, prompt_tokens: int, completion_tokens: int
    ) -> dict:
        pin, pout = self._price_for(spec)
        cost = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
        self.calls += 1
        self.total_cost_usd += cost
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        slot = self.by_provider.setdefault(
            spec, {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0,
                   "completion_tokens": 0},
        )
        slot["calls"] += 1
        slot["cost_usd"] += cost
        slot["prompt_tokens"] += prompt_tokens
        slot["completion_tokens"] += completion_tokens
        payload = {
            "provider": spec, "call_cost_usd": round(cost, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "calls": self.calls,
        }
        if self.on_cost is not None:
            try:
                self.on_cost(payload)
            except Exception:
                logger.exception("on_cost callback raised")
        return payload

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "spend_cap_usd": self.spend_cap_usd,
            "calls": self.calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_provider": self.by_provider,
        }


def _usage_from_response(response: Any) -> Optional[tuple[int, int]]:
    """Pull real (prompt, completion) token usage off a LangChain response."""
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict) and "input_tokens" in meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(usage, dict) and usage:
            pin = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            pout = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            return int(pin), int(pout)
    return None
