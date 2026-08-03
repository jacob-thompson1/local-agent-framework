"""Tool registration, token-cost accounting, relevance pruning, and analysis.

Every tool carries:

* a machine-readable schema derived from the function signature,
* a **measured token cost** (what its definition actually consumes in the
  prompt, computed with the active :class:`~my_agent_framework.tokens.TokenCounter`),
* a :class:`~my_agent_framework.safety.Severity` classification used by the
  approval layer,
* optional keywords used by the offline relevance pruner.

Pruning is deliberately dependency-free (keyword/substring scoring, no
embeddings, no network) so it works on air-gapped machines. It is a coarse
filter -- the docs are explicit that for 3B models you should usually pass an
explicit tool subset rather than relying on automatic selection.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, get_type_hints

from .safety import Severity
from .tokens import TokenCounter

logger = logging.getLogger("my_agent_framework.registry")

_PY_TO_JSON = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
}


def _json_type(annotation: Any) -> str:
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        annotation = origin
    return _PY_TO_JSON.get(annotation, "string")


@dataclass
class ToolSpec:
    """A registered tool: callable + metadata + measured token cost."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    severity: Severity = Severity.READ_ONLY
    keywords: list[str] = field(default_factory=list)
    timeout_s: Optional[float] = None      # per-tool override of the default
    token_cost: int = 0                    # measured at registration

    def render_definition(self) -> str:
        """The exact text injected into the prompt for this tool.

        Kept terse on purpose: name, one-line description, compact JSON-ish
        parameter block. This string is what ``token_cost`` measures.
        """
        params = ", ".join(
            f'"{p}": {meta["type"]}' + ("" if p in self.required else "?")
            for p, meta in self.parameters.items()
        )
        return f"- {self.name}({params}): {self.description}"

    def to_schema(self) -> dict:
        """OpenAI-style JSON schema (for audit logs and interop)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    p: {"type": m["type"], "description": m.get("description", "")}
                    for p, m in self.parameters.items()
                },
                "required": self.required,
            },
            "severity": self.severity.value,
            "token_cost": self.token_cost,
        }


def tool(
    description: str,
    *,
    severity: Severity = Severity.READ_ONLY,
    keywords: Optional[Iterable[str]] = None,
    timeout_s: Optional[float] = None,
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator turning a plain Python function into a registrable tool.

    Example
    -------
    >>> @tool("Add two numbers.", keywords=["math", "sum"])
    ... def add(a: float, b: float) -> float:
    ...     return a + b

    Parameter descriptions can be given via a ``param_docs`` dict attribute or
    are left blank (blank is fine for small models -- shorter is better).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__tool_meta__ = {  # type: ignore[attr-defined]
            "description": description.strip(),
            "severity": severity,
            "keywords": list(keywords or []),
            "timeout_s": timeout_s,
            "name": name or func.__name__,
        }
        return func

    return decorator


def _spec_from_func(func: Callable[..., Any], counter: TokenCounter) -> ToolSpec:
    meta = getattr(func, "__tool_meta__", None)
    if meta is None:
        raise TypeError(
            f"{func.__name__} is not a tool. Decorate it with "
            "@my_agent_framework.tool('description', ...) first."
        )
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    parameters: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    param_docs: dict[str, str] = getattr(func, "param_docs", {})
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        parameters[pname] = {
            "type": _json_type(hints.get(pname, str)),
            "description": param_docs.get(pname, ""),
        }
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    spec = ToolSpec(
        name=meta["name"],
        description=meta["description"],
        func=func,
        parameters=parameters,
        required=required,
        severity=meta["severity"],
        keywords=meta["keywords"],
        timeout_s=meta["timeout_s"],
    )
    spec.token_cost = counter.count(spec.render_definition())
    return spec


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z_]{3,}")


class ToolRegistry:
    """Holds registered tools; selects a relevant, budget-respecting subset.

    The registry is *global inventory*; each agent session gets a pruned
    *active set* chosen by :meth:`select_for_task`.
    """

    def __init__(self, counter: Optional[TokenCounter] = None) -> None:
        self.counter = counter or TokenCounter()
        self._tools: dict[str, ToolSpec] = {}

    # -- registration ------------------------------------------------------

    def register(self, func: Callable[..., Any]) -> ToolSpec:
        spec = _spec_from_func(func, self.counter)
        if spec.name in self._tools:
            logger.warning("Tool %r re-registered; overwriting.", spec.name)
        self._tools[spec.name] = spec
        logger.info(
            "Registered tool %r (severity=%s, ~%d prompt tokens)",
            spec.name, spec.severity.value, spec.token_cost,
        )
        return spec

    def register_all(self, funcs: Iterable[Callable[..., Any]]) -> list[ToolSpec]:
        return [self.register(f) for f in funcs]

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    # -- pruning -----------------------------------------------------------

    def select_for_task(
        self,
        task: str,
        max_tools: int,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> list[ToolSpec]:
        """Choose <= *max_tools* tools relevant to *task*.

        Scoring is offline keyword overlap between the task text and each
        tool's name/description/keywords. Tools named in *include* are always
        kept (and count against the budget first); *exclude* wins over
        everything. If the registry already fits the budget, no pruning occurs.
        """
        include_set = set(include or [])
        exclude_set = set(exclude or [])
        candidates = [t for t in self._tools.values() if t.name not in exclude_set]

        if len(candidates) <= max_tools:
            return candidates

        task_words = {w.lower() for w in _WORD_RE.findall(task)}

        def score(spec: ToolSpec) -> float:
            corpus = " ".join(
                [spec.name.replace("_", " "), spec.description, " ".join(spec.keywords)]
            ).lower()
            corpus_words = set(_WORD_RE.findall(corpus))
            overlap = len(task_words & corpus_words)
            # substring bonus: task mentions the tool name directly
            bonus = 2.0 if spec.name.lower() in task.lower() else 0.0
            # cheap tools break ties (lower cost slightly preferred)
            return overlap + bonus - spec.token_cost / 1000.0

        pinned = [t for t in candidates if t.name in include_set]
        rest = sorted(
            (t for t in candidates if t.name not in include_set),
            key=score, reverse=True,
        )
        selected = (pinned + rest)[:max_tools]
        dropped = [t.name for t in candidates if t not in selected]
        if dropped:
            logger.info(
                "Tool pruning: kept %s; dropped %s (budget=%d). "
                "Saved ~%d prompt tokens.",
                [t.name for t in selected], dropped, max_tools,
                sum(self._tools[d].token_cost for d in dropped),
            )
        return selected

    # -- analysis ----------------------------------------------------------

    def analyze(
        self,
        size_class: str,
        context_window: int,
        system_prompt_tokens: int = 0,
        tool_names: Optional[Iterable[str]] = None,
    ) -> dict:
        """Configuration weight report: is this tool set sane for this model?

        Returns a dict with per-tool costs, totals, and human-readable
        warnings. Used by the CLI ``my-agent analyze`` command and available
        programmatically.
        """
        from .models import PROFILES  # noqa: PLC0415 (avoid cycle at import)

        profile = PROFILES.get(size_class, PROFILES["7b"])
        tools = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names is not None else self.all()
        )
        per_tool = [
            {"name": t.name, "tokens": t.token_cost, "severity": t.severity.value}
            for t in sorted(tools, key=lambda t: -t.token_cost)
        ]
        total_tool_tokens = sum(t.token_cost for t in tools)
        warnings: list[str] = []
        if len(tools) > profile.max_tools:
            warnings.append(
                f"{len(tools)} tools exceeds the {profile.max_tools}-tool "
                f"recommendation for {size_class} models. Small models pick the "
                "wrong tool more often as the menu grows -- prune to "
                f"{profile.recommended_tools} or pass an explicit subset per task."
            )
        overhead = system_prompt_tokens + total_tool_tokens
        if overhead > context_window * 0.35:
            warnings.append(
                f"System prompt + tool definitions consume ~{overhead} tokens "
                f"({overhead / context_window:.0%} of a {context_window}-token "
                "window) before any conversation happens. Aim for <35%."
            )
        heavy = [t for t in tools if t.token_cost > 80]
        for t in heavy:
            warnings.append(
                f"Tool '{t.name}' costs ~{t.token_cost} tokens -- consider a "
                "shorter description (target <=50)."
            )
        return {
            "size_class": size_class,
            "profile_max_tools": profile.max_tools,
            "tool_count": len(tools),
            "per_tool_tokens": per_tool,
            "total_tool_tokens": total_tool_tokens,
            "system_prompt_tokens": system_prompt_tokens,
            "context_window": context_window,
            "fixed_overhead_fraction": round(
                (system_prompt_tokens + total_tool_tokens) / max(context_window, 1), 3
            ),
            "warnings": warnings,
            "ok": not warnings,
        }
