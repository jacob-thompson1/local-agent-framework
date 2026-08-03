"""my-agent-framework: a small-model-first agent framework on LangChain.

Importing this package triggers **zero network activity** -- provider SDKs
load lazily, and no telemetry, version checks, or downloads ever occur.

Quick start::

    from my_agent_framework import SmallModelAgent, tool

    @tool("Add two numbers.")
    def add(a: float, b: float) -> float:
        return a + b

    agent = SmallModelAgent("ollama:mistral:7b", tools=[add])
    result = agent.run("What is 41.5 + 0.5?")
    print(result.final_answer)
"""

from .agent import AgentResult, AgentStep, SmallModelAgent, __version__
from .checkpoint import Checkpoint, WorkspaceCheckpointer
from .editing import EditSession, ProposedEdit, flexible_find
from .events import EventBus, QueueConfirmer
from .audit import AuditExporter, AuditLogger, DEFAULT_REDACTION_PATTERNS
from .config import Settings, agent_from_settings
from .memory import ConversationMemory
from .models import (
    CostTracker, SpendCapExceeded,
    FAMILY_HINTS,
    PROFILES,
    HybridChatModel,
    ModelProfile,
    build_chat_model,
    detect_size_class,
)
from .registry import ToolRegistry, ToolSpec, tool
from .safety import (
    PermissionPolicy,
    Severity,
    paranoid_policy,
    read_only_policy,
    stdin_confirmer,
)
from .tokens import DEFAULT_PRICING, PRICING_AS_OF, estimate_cost, lookup_price
from .tokens import ContextBudget, TokenCounter

__all__ = [
    "Checkpoint", "WorkspaceCheckpointer", "EditSession", "ProposedEdit",
    "flexible_find", "EventBus", "QueueConfirmer", "CostTracker",
    "SpendCapExceeded", "DEFAULT_PRICING", "PRICING_AS_OF", "estimate_cost",
    "lookup_price",
    "__version__",
    "SmallModelAgent", "AgentResult", "AgentStep",
    "tool", "ToolRegistry", "ToolSpec",
    "Severity", "PermissionPolicy", "read_only_policy", "paranoid_policy",
    "stdin_confirmer",
    "AuditLogger", "AuditExporter", "DEFAULT_REDACTION_PATTERNS",
    "Settings", "agent_from_settings",
    "HybridChatModel", "build_chat_model", "detect_size_class",
    "ModelProfile", "PROFILES", "FAMILY_HINTS",
    "TokenCounter", "ContextBudget", "ConversationMemory",
]
