"""Safety modes and the audit trail, end to end.

Shows: confirmation mode, a read-only agent, dry-run planning, and exporting
the session's audit log with PII redaction.
"""

from __future__ import annotations

import json

from my_agent_framework import (
    AuditExporter,
    DEFAULT_REDACTION_PATTERNS,
    PermissionPolicy,
    Severity,
    SmallModelAgent,
    read_only_policy,
)
from my_agent_framework.tools import FULL_TOOLS

MODEL = "ollama:mistral:7b-instruct"

# 1. "This agent will execute code -- require confirmation."
#    run_python is DESTRUCTIVE, write_file is WRITE; both prompt on stdin.
confirming_agent = SmallModelAgent(
    MODEL, tools=FULL_TOOLS,
    policy=PermissionPolicy(confirm_at_or_above=Severity.WRITE),
)

# 2. "This agent can only read files, no writes allowed."
#    WRITE/DESTRUCTIVE tools are hard-blocked; the model is told the action
#    was blocked and routes around it.
readonly_agent = SmallModelAgent(MODEL, tools=FULL_TOOLS, policy=read_only_policy())

# 3. Dry-run: nothing executes; the result contains the planned actions.
planner = SmallModelAgent(
    MODEL, tools=FULL_TOOLS, policy=PermissionPolicy(dry_run=True),
)

# 4. Custom confirmer for non-terminal apps (Streamlit, web UI, etc.):
def my_confirmer(tool_name: str, args: dict, severity: Severity, reason: str) -> bool:
    # Replace with a UI dialog / approval queue in real apps.
    print(f"Approving {severity.value} call to {tool_name}? -> auto-yes (demo)")
    return True

ui_agent = SmallModelAgent(
    MODEL, tools=FULL_TOOLS,
    policy=PermissionPolicy(confirm_at_or_above=Severity.WRITE,
                            confirmer=my_confirmer),
)

if __name__ == "__main__":
    result = planner.run("Save a haiku about autumn to haiku.txt")
    print(result.final_answer)          # [DRY RUN] Planned actions ...

    # Export this session's audit log, redacted, for review:
    exporter = AuditExporter(redact_patterns=DEFAULT_REDACTION_PATTERNS)
    print(json.dumps(
        json.loads(exporter.export("json", session_id=result.session_id))[:3],
        indent=2,
    ))
