"""SmallModelAgent: a lean, auditable reasoning loop for 3B-7B models.

Why not LangChain's stock agent executors? Two reasons that matter at small
scale:

1. Native function-calling is unreliable or absent on many small local models.
   This agent uses a **strict single-JSON-object protocol** that every
   instruction-tuned 3B+ model can follow, with corrective retries when the
   model emits malformed output.
2. Stock prompts are verbose. The system prompt here is ~120 tokens plus
   ~15-50 tokens per tool -- every token is measured and reported.

The agent still builds on LangChain: any ``langchain_core`` ``BaseChatModel``
plugs in unchanged, messages use ``langchain_core.messages``, and tools can be
exported as OpenAI-style schemas for interop.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Union

from .audit import AuditLogger
from .checkpoint import WorkspaceCheckpointer
from .events import EventBus
from .memory import ConversationMemory
from .models import (
    FAMILY_HINTS, PROFILES, CostTracker, HybridChatModel,
    ModelProfile, build_chat_model, detect_family, detect_size_class,
)
from .registry import ToolRegistry, ToolSpec
from .safety import PermissionPolicy, Severity
from .tokens import ContextBudget, TokenCounter

logger = logging.getLogger("my_agent_framework.agent")

__version__ = "0.2.0"

#: arg names treated as filesystem paths for workspace confinement
_PATHY_ARG_NAMES = ("path", "file", "filename", "filepath", "dir", "directory",
                    "dest", "destination", "source", "src", "target")

# ---------------------------------------------------------------------------
# Prompt construction (kept deliberately terse)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """You are a precise assistant that solves tasks using tools.

TOOLS:
{tool_block}

OUTPUT RULES:
Reply with EXACTLY one JSON object and nothing else.
To use a tool:
{{"thought": "<brief reasoning>", "tool": "<tool name>", "args": {{<parameters>}}}}
To give the final answer:
{{"thought": "<brief reasoning>", "final": "<answer>"}}
Use only listed tools. Use "final" as soon as you can answer.{few_shot}"""

_FEW_SHOT = """

EXAMPLE:
Task: What is 12 * 7?
{"thought": "Use the calculator.", "tool": "calculator", "args": {"expression": "12*7"}}
Observation: 84
{"thought": "I have the result.", "final": "12 * 7 = 84"}"""

_PLAN_TEMPLATE = """You plan tasks. Do NOT solve the task or call tools.
Reply with EXACTLY one JSON object:
{{"plan": ["<step 1>", "<step 2>", ...]}}
3 steps or fewer, under 10 words each.
Available tools: {tool_names}
Task: {task}"""

_CORRECTION_MSG = (
    'Your reply was not a single valid JSON object. Reply again with ONLY one '
    'JSON object: {"thought": "...", "tool": "...", "args": {...}} or '
    '{"thought": "...", "final": "..."}.'
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[dict]:
    """Pull the first parseable JSON object out of model output.

    Tolerates markdown fences, leading narration, and trailing prose -- the
    usual small-model failure modes. Returns None if nothing parses.
    """
    text = text.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # fast path
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # scan for balanced braces starting at each '{'
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
    return None


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One iteration of the reasoning loop."""

    iteration: int
    thought: str = ""
    tool: Optional[str] = None
    args: Optional[dict] = None
    permission_outcome: Optional[str] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


@dataclass
class AgentResult:
    """Structured outcome of :meth:`SmallModelAgent.run`."""

    # "success" | "max_iterations" | "timeout" | "error" | "dry_run"
    # | "plan_rejected"
    status: str
    final_answer: Optional[str]
    steps: list[AgentStep] = field(default_factory=list)
    session_id: str = ""
    audit_path: Optional[str] = None
    iterations: int = 0
    total_prompt_tokens: int = 0
    model_used: str = ""
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "final_answer": self.final_answer,
            "session_id": self.session_id,
            "audit_path": self.audit_path,
            "iterations": self.iterations,
            "total_prompt_tokens": self.total_prompt_tokens,
            "model_used": self.model_used,
            "error": self.error,
            "steps": [s.as_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class SmallModelAgent:
    """Tool-using agent tuned for small local models.

    Parameters
    ----------
    llm:
        One of: a provider string (``"ollama:mistral:7b"``), any object with a
        LangChain-style ``.invoke(messages)`` (``BaseChatModel``), or a
        :class:`HybridChatModel`. Provider strings are resolved lazily -- no
        network activity at construction.
    tools:
        A :class:`ToolRegistry` or an iterable of ``@tool``-decorated
        functions.
    size_class:
        ``"3b" | "5b" | "7b" | "large"``. Auto-detected from the model name if
        omitted; drives default tool limits, iteration caps, and few-shot use.
    policy:
        :class:`PermissionPolicy` controlling confirmation/dry-run/blocking.
    context_window / max_iterations / few_shot:
        Override the profile defaults.
    tool_timeout_s / session_timeout_s:
        Per-tool call timeout (default 30s) and whole-run timeout (default
        300s = 5 minutes).
    audit_enabled / audit_root / user / role:
        Compliance logging controls. See :mod:`my_agent_framework.audit`.
    merge_system_into_user:
        For serving stacks that mishandle system messages (some Gemma
        deployments): prepend the system prompt to the first user message.
    llm_retries:
        Transient-failure retries per LLM call (exponential backoff).
    sensitive_task:
        Flag this session's decisions for bias/fairness review in the audit
        log (e.g. anything touching underwriting or claims outcomes).
    """

    def __init__(
        self,
        llm: Union[str, HybridChatModel, Any],
        tools: Union[ToolRegistry, Iterable[Callable[..., Any]], None] = None,
        *,
        size_class: Optional[str] = None,
        policy: Optional[PermissionPolicy] = None,
        context_window: Optional[int] = None,
        max_iterations: Optional[int] = None,
        max_tools: Optional[int] = None,
        few_shot: Optional[bool] = None,
        memory_tokens: Optional[int] = None,
        tool_timeout_s: float = 30.0,
        session_timeout_s: float = 300.0,
        audit_enabled: bool = True,
        audit_root: Optional[Any] = None,
        user: str = "unknown",
        role: str = "unknown",
        merge_system_into_user: bool = False,
        llm_retries: int = 2,
        sensitive_task: bool = False,
        model_kwargs: Optional[dict] = None,
        bus: Optional[EventBus] = None,
        plan_first: bool = False,
        workspace_root: Optional[str] = None,
        checkpointer: Optional[WorkspaceCheckpointer] = None,
        cost_tracker: Optional[CostTracker] = None,
        edit_session: Optional[Any] = None,
    ) -> None:
        self._llm_input = llm
        self._llm: Optional[Any] = None
        self._model_kwargs = model_kwargs or {}
        self.model_name = self._infer_model_name(llm)

        self.size_class = size_class or detect_size_class(self.model_name)
        self.profile: ModelProfile = PROFILES.get(self.size_class, PROFILES["7b"])
        family = detect_family(self.model_name)
        if family:
            logger.info("Model family hint (%s): %s", family, FAMILY_HINTS[family])

        self.counter = TokenCounter()
        if isinstance(tools, ToolRegistry):
            self.registry = tools
            self.registry.counter = self.counter
        else:
            self.registry = ToolRegistry(self.counter)
            if tools:
                self.registry.register_all(tools)

        self.policy = policy or PermissionPolicy()
        self.context_window = context_window or self.profile.context_window
        self.max_iterations = max_iterations or self.profile.max_iterations
        self.max_tools = max_tools or self.profile.max_tools
        self.few_shot = (
            few_shot if few_shot is not None else self.profile.few_shot_examples > 0
        )
        self.tool_timeout_s = tool_timeout_s
        self.session_timeout_s = session_timeout_s
        self.merge_system_into_user = merge_system_into_user
        self.llm_retries = llm_retries
        self.sensitive_task = sensitive_task

        self.budget = ContextBudget(
            context_window=self.context_window,
            counter=self.counter,
        )
        self.memory = ConversationMemory(
            max_tokens=memory_tokens or max(512, int(self.context_window * 0.4)),
            counter=self.counter,
        )

        self._audit_enabled = audit_enabled
        self._audit_root = audit_root
        self._user = user
        self._role = role

        # -- v0.2: events, planning, workspace, cost ------------------------
        self.bus = bus or EventBus()
        self.plan_first = plan_first
        if plan_first and self.size_class in ("3b", "5b"):
            logger.warning(
                "plan_first with a %s model: small models plan poorly and the "
                "pinned plan consumes context all session. Consider a 7b+ "
                "model or plan_first=False.", self.size_class,
            )
        self.workspace_root = (
            None if workspace_root is None else str(workspace_root)
        )
        if checkpointer is not None:
            self.checkpointer = checkpointer
            self.workspace_root = str(checkpointer.workspace)
        elif self.workspace_root is not None:
            self.checkpointer = WorkspaceCheckpointer(self.workspace_root)
        else:
            self.checkpointer = None
        self.cost_tracker = cost_tracker
        if (
            self.cost_tracker is not None
            and isinstance(llm, HybridChatModel)
        ):
            llm.cost_tracker = self.cost_tracker
        if self.cost_tracker is not None:
            self.cost_tracker.on_cost = lambda p: self.bus.emit("cost_update", p)
        if edit_session is not None:
            self.registry.register_all(edit_session.tools())
            edit_session.on_propose = (
                lambda p: self.bus.emit("diff_proposed", p)
            )
            if edit_session.workspace_root is None and self.workspace_root:
                from pathlib import Path
                edit_session.workspace_root = Path(self.workspace_root).resolve()
        self.edit_session = edit_session

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _infer_model_name(llm: Any) -> str:
        if isinstance(llm, str):
            return llm
        if isinstance(llm, HybridChatModel):
            return llm.model_name
        for attr in ("model", "model_name", "model_id"):
            value = getattr(llm, attr, None)
            if isinstance(value, str):
                return value
        return llm.__class__.__name__

    def _get_llm(self) -> Any:
        if self._llm is None:
            if isinstance(self._llm_input, str):
                self._llm = build_chat_model(self._llm_input, **self._model_kwargs)
            else:
                self._llm = self._llm_input
        return self._llm

    def _build_system_prompt(self, tools: list[ToolSpec]) -> str:
        tool_block = "\n".join(t.render_definition() for t in tools) or "(none)"
        few_shot = _FEW_SHOT if self.few_shot else ""
        return _SYSTEM_TEMPLATE.format(tool_block=tool_block, few_shot=few_shot)

    def _invoke_llm(self, system: str, history: str, audit: AuditLogger) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage  # lazy

        if self.merge_system_into_user:
            messages: list[Any] = [HumanMessage(content=f"{system}\n\n{history}")]
        else:
            messages = [SystemMessage(content=system), HumanMessage(content=history)]

        last_exc: Optional[Exception] = None
        for attempt in range(self.llm_retries + 1):
            try:
                response = self._get_llm().invoke(messages)
                content = getattr(response, "content", response)
                if isinstance(content, list):  # multimodal-style content blocks
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                return str(content)
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                audit.error(
                    "llm_invoke", f"{type(exc).__name__}: {exc}",
                    recovery=f"retry {attempt + 1}/{self.llm_retries} after {wait}s"
                    if attempt < self.llm_retries else "giving up",
                )
                if attempt < self.llm_retries:
                    logger.warning(
                        "LLM call failed (%s); retrying in %ds (%d/%d)",
                        exc, wait, attempt + 1, self.llm_retries,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM invocation failed after retries: {last_exc}"
        ) from last_exc

    def _execute_tool(self, spec: ToolSpec, args: dict) -> Any:
        timeout = spec.timeout_s or self.tool_timeout_s
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(spec.func, **args)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Tool '{spec.name}' exceeded its {timeout:.0f}s timeout."
                )

    def _confinement_error(self, spec: ToolSpec, args: dict) -> Optional[str]:
        """Workspace confinement for WRITE/DESTRUCTIVE tools.

        Checks args whose *names* look path-like (path, file, dir, dest, ...).
        Heuristic by design -- custom tools with unusual arg names should
        validate paths themselves (see EditSession, which confines properly).
        """
        if self.workspace_root is None or spec.severity == Severity.READ_ONLY:
            return None
        from pathlib import Path
        root = Path(self.workspace_root).resolve()
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            if not any(p in key.lower() for p in _PATHY_ARG_NAMES):
                continue
            candidate = Path(value).expanduser()
            resolved = (
                candidate if candidate.is_absolute() else root / candidate
            ).resolve()
            if not resolved.is_relative_to(root):
                return (
                    f"Blocked: '{value}' is outside the workspace "
                    f"'{root}'. Use a path inside the workspace."
                )
        return None

    def _make_plan(
        self, task: str, active: list[ToolSpec], audit: AuditLogger,
    ) -> tuple[Optional[list[str]], Optional[str]]:
        """One tool-less LLM call -> (steps, None) or (None, abort_reason)."""
        prompt = _PLAN_TEMPLATE.format(
            tool_names=", ".join(t.name for t in active) or "(none)", task=task,
        )
        raw = self._invoke_llm(prompt, f"Task: {task}", audit)
        parsed = extract_json_object(raw)
        steps: list[str] = []
        if parsed and isinstance(parsed.get("plan"), list):
            steps = [str(s) for s in parsed["plan"][:3]]
        if not steps:
            # A bad plan is not worth an extra corrective call; proceed
            # planless rather than burn budget (act phase is unaffected).
            audit.event("plan", steps=[], approved=None,
                        note="unparseable plan; proceeding without one")
            return None, None
        # hard cap the pinned cost (~80 tokens)
        while len(steps) > 1 and self.counter.count("; ".join(steps)) > 80:
            steps.pop()
        approved: Optional[bool] = None
        if self.policy.confirm_at_or_above is not None:
            decision = self.policy.check(
                "execution_plan", {"plan": steps}, Severity.WRITE,
                "approve the plan before execution",
            )
            if decision.outcome not in ("allowed", "approved", "dry_run"):
                audit.event("plan", steps=steps, approved=False,
                            outcome=decision.outcome)
                return None, f"Plan {decision.outcome} by user/policy."
            approved = decision.outcome == "approved"
        audit.event("plan", steps=steps, approved=approved)
        return steps, None

    # -- public API --------------------------------------------------------

    def run(self, task: str, tool_subset: Optional[Iterable[str]] = None) -> AgentResult:
        """Execute the reasoning loop for *task* and return a structured result.

        ``tool_subset`` optionally names the exact tools to expose this
        session (recommended for 3B models); otherwise the registry prunes to
        the profile's tool budget automatically.
        """
        audit = AuditLogger(
            user=self._user, role=self._role, enabled=self._audit_enabled,
            bus=self.bus,
            **({"root": self._audit_root} if self._audit_root else {}),
        )
        self.memory.on_compaction = lambda entry: audit.event(
            "compaction", **entry
        )
        if isinstance(self._llm_input, HybridChatModel):
            if self.cost_tracker is not None:
                self._llm_input.cost_tracker = self.cost_tracker
            self._llm_input.on_fallback = lambda exc: audit.fallback(
                self._llm_input.primary_spec,
                self._llm_input.fallback_spec or "?",
                f"{type(exc).__name__}: {exc}",
            )

        if tool_subset is not None:
            active = [t for t in (self.registry.get(n) for n in tool_subset) if t]
        else:
            active = self.registry.select_for_task(task, self.max_tools)
        tools_by_name = {t.name: t for t in active}
        system = self._build_system_prompt(active)

        audit.session_start(
            model=self.model_name,
            model_params={
                "size_class": self.size_class,
                "context_window": self.context_window,
                "max_iterations": self.max_iterations,
                "tool_timeout_s": self.tool_timeout_s,
                "session_timeout_s": self.session_timeout_s,
            },
            tools=[t.to_schema() for t in active],
            framework_version=__version__,
            task=task,
            config={
                "dry_run": self.policy.dry_run,
                "confirm_at_or_above": (
                    self.policy.confirm_at_or_above.value
                    if self.policy.confirm_at_or_above else None
                ),
                "max_severity": self.policy.max_severity.value,
                "few_shot": self.few_shot,
            },
        )
        logger.info(
            "Session %s: model=%s size=%s tools=%s (%d prompt tokens for tools, "
            "%d for system prompt)",
            audit.session_id, self.model_name, self.size_class,
            list(tools_by_name), sum(t.token_cost for t in active),
            self.counter.count(system),
        )

        self.memory.clear()
        self.budget.reset_warnings()
        self.memory.add("user", f"Task: {task}", pinned=True)

        if self.plan_first:
            steps, abort_reason = self._make_plan(task, active, audit)
            if abort_reason is not None:
                result = AgentResult(
                    status="plan_rejected", final_answer=None,
                    session_id=audit.session_id,
                    audit_path=str(audit.path) if self._audit_enabled else None,
                    model_used=self.model_name, error=abort_reason,
                )
                audit.session_end(result.status, None, 0, 0)
                return result
            if steps:
                self.memory.add(
                    "user", "Plan: " + "; ".join(steps), pinned=True
                )

        result = AgentResult(
            status="error", final_answer=None, session_id=audit.session_id,
            audit_path=str(audit.path) if self._audit_enabled else None,
            model_used=self.model_name,
        )
        started = time.monotonic()
        parse_failures = 0
        dry_run_plan: list[str] = []

        try:
            for iteration in range(1, self.max_iterations + 1):
                if time.monotonic() - started > self.session_timeout_s:
                    result.status = "timeout"
                    result.error = (
                        f"Session exceeded {self.session_timeout_s:.0f}s limit."
                    )
                    audit.error("session", result.error, recovery="stopped cleanly")
                    break

                history = self.memory.render()
                report = self.budget.report(system_text=system, history_texts=[history])
                result.total_prompt_tokens += report.used

                step = AgentStep(iteration=iteration)
                t0 = time.monotonic()
                raw = self._invoke_llm(system, history, audit)
                audit.llm_call(iteration, report.used, raw)

                parsed = extract_json_object(raw)
                if parsed is None:
                    parse_failures += 1
                    audit.error(
                        "parse", "model output was not a valid JSON object",
                        recovery="corrective retry" if parse_failures <= 2 else "abort",
                    )
                    if parse_failures > 2:
                        result.status = "error"
                        result.error = "Model repeatedly produced unparseable output."
                        break
                    self.memory.add("assistant", raw[:400])
                    self.memory.add("user", _CORRECTION_MSG)
                    continue

                step.thought = str(parsed.get("thought", ""))

                # -- final answer --------------------------------------------
                if "final" in parsed:
                    audit.decision(iteration, step.thought, None, None,
                                   sensitive=self.sensitive_task)
                    step.duration_s = time.monotonic() - t0
                    result.steps.append(step)
                    result.final_answer = str(parsed["final"])
                    result.status = "dry_run" if self.policy.dry_run and dry_run_plan \
                        else "success"
                    break

                # -- tool call -----------------------------------------------
                tool_name = parsed.get("tool")
                args = parsed.get("args") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                step.tool = tool_name
                step.args = args
                audit.decision(iteration, step.thought, tool_name, args,
                               sensitive=self.sensitive_task)

                spec = tools_by_name.get(str(tool_name))
                if spec is None:
                    observation = (
                        f"Error: '{tool_name}' is not an available tool. "
                        f"Available: {sorted(tools_by_name)}."
                    )
                    step.error = observation
                elif (confine_err := self._confinement_error(spec, args)):
                    observation = confine_err
                    step.error = observation
                    step.permission_outcome = "blocked"
                    audit.approval(
                        spec.name, spec.severity.value, "blocked", False,
                        "workspace confinement",
                    )
                else:
                    decision = self.policy.check(
                        spec.name, args, spec.severity, step.thought
                    )
                    step.permission_outcome = decision.outcome
                    audit.approval(
                        spec.name, spec.severity.value, decision.outcome,
                        decision.required_approval, decision.detail,
                    )
                    if decision.outcome == "dry_run":
                        dry_run_plan.append(f"{spec.name}({args})")
                        observation = (
                            f"[dry-run] '{spec.name}' was NOT executed. Assume it "
                            "would succeed and continue planning, or finish."
                        )
                    elif not decision.may_execute:
                        observation = (
                            f"Action '{spec.name}' was {decision.outcome} by the "
                            "user/policy. Choose a different approach or finish."
                        )
                    else:
                        exec_t0 = time.monotonic()
                        try:
                            value = self._execute_tool(spec, args)
                            observation = str(value)
                            audit.tool_result(
                                spec.name, True, observation,
                                time.monotonic() - exec_t0,
                            )
                            if (
                                self.checkpointer is not None
                                and spec.severity != Severity.READ_ONLY
                            ):
                                try:
                                    cp = self.checkpointer.checkpoint(
                                        label=f"after {spec.name} (iter {iteration})",
                                        memory_turns=len(self.memory.turns),
                                    )
                                    audit.event("checkpoint", **cp.as_dict())
                                except Exception:
                                    logger.exception(
                                        "Checkpoint failed; continuing "
                                        "(execution result stands)."
                                    )
                        except Exception as exc:
                            observation = f"Error running '{spec.name}': {exc}"
                            step.error = observation
                            audit.tool_result(
                                spec.name, False, None,
                                time.monotonic() - exec_t0, error=str(exc),
                            )

                step.observation = observation[:2000]
                step.duration_s = time.monotonic() - t0
                result.steps.append(step)
                self.memory.add("assistant", json.dumps(parsed, ensure_ascii=False))
                self.memory.add("tool", step.observation)
            else:
                result.status = "max_iterations"
                result.error = (
                    f"No final answer after {self.max_iterations} iterations."
                )
        except Exception as exc:
            logger.exception("Agent run failed")
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            audit.error("agent_loop", result.error, recovery="session aborted")

        if isinstance(self._llm_input, HybridChatModel):
            result.model_used = self._llm_input.last_provider_used or self.model_name

        result.iterations = len(result.steps)
        if result.status == "dry_run" and dry_run_plan:
            plan = "; ".join(dry_run_plan)
            result.final_answer = (
                f"[DRY RUN] Planned actions (not executed): {plan}. "
                f"Model's conclusion: {result.final_answer}"
            )
        audit.session_end(
            result.status, result.final_answer, result.iterations,
            result.total_prompt_tokens,
        )
        return result
