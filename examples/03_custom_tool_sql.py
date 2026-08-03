"""Adding a domain-specific tool: read-only SQL against SQLite.

Demonstrates the full custom-tool pattern:
  1. write a plain function with type hints,
  2. decorate with @tool (description, severity, keywords, timeout),
  3. register it -- the framework measures its prompt token cost automatically,
  4. flag the session `sensitive_task=True` so every decision is marked for
     bias/fairness review in the audit log (relevant when queries could touch
     protected-class data in insurance workflows).

Error handling rule: raise exceptions freely inside a tool. The agent catches
them, feeds the message back to the model as an observation, and logs the
failure -- small models are surprisingly good at correcting a bad SQL query
when shown the database error.
"""

from __future__ import annotations

import sqlite3

from my_agent_framework import (
    PermissionPolicy, Severity, SmallModelAgent, ToolRegistry, tool,
)

DB_PATH = "claims_demo.db"


def _setup_demo_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY, program TEXT, amount REAL, status TEXT
        );
        DELETE FROM claims;
        INSERT INTO claims (program, amount, status) VALUES
            ('AUTO-A', 1200.50, 'paid'), ('AUTO-A', 890.00, 'open'),
            ('PROP-B', 15400.00, 'paid'), ('PROP-B', 2300.75, 'denied');
        """
    )
    conn.commit()
    conn.close()


@tool(
    "Run a read-only SQL SELECT against the claims database. "
    "Tables: claims(id, program, amount, status).",
    severity=Severity.READ_ONLY,
    keywords=["sql", "query", "database", "claims", "select", "table"],
    timeout_s=10.0,
)
def query_claims(sql: str) -> str:
    """Read-only enforced two ways: statement whitelist + SQLite URI ro mode."""
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cursor = conn.execute(sql)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchmany(50)
        lines = [" | ".join(cols)]
        lines += [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)
    finally:
        conn.close()


def main() -> None:
    _setup_demo_db()

    registry = ToolRegistry()
    spec = registry.register(query_claims)
    print(f"query_claims costs ~{spec.token_cost} prompt tokens\n")

    agent = SmallModelAgent(
        "ollama:qwen2.5:7b",
        tools=registry,
        policy=PermissionPolicy(confirm_at_or_above=Severity.WRITE),
        sensitive_task=True,   # decisions flagged for fairness review in audit
        user="jacob", role="analyst",
    )
    result = agent.run("What is the total paid claim amount per program?")
    print(result.final_answer)
    print("audit:", result.audit_path)


if __name__ == "__main__":
    main()
