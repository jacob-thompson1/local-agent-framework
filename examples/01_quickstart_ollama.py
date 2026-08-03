"""Quickstart: local Ollama models at 3B, 5B-class, and 7B.

Prereqs:
    pip install my-agent-framework[ollama]
    ollama pull qwen2.5:3b      # and/or phi3:mini, mistral:7b-instruct

Run:
    python 01_quickstart_ollama.py [3b|5b|7b]
"""

from __future__ import annotations

import logging
import sys

from my_agent_framework import PermissionPolicy, Severity, SmallModelAgent
from my_agent_framework.tools import BASIC_TOOLS, FULL_TOOLS, STANDARD_TOOLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Recommended configurations by size. Note how tool count, iterations, and
# memory budget all shrink together as the model shrinks.
CONFIGS = {
    "3b": dict(
        model="ollama:qwen2.5:3b",     # Qwen is the strongest 3B JSON emitter
        tools=BASIC_TOOLS,             # 3 tools, ~60 prompt tokens
        kwargs=dict(size_class="3b", max_iterations=5, context_window=4096),
    ),
    "5b": dict(
        model="ollama:phi3:mini",      # ~3.8B but punches at 5B class
        tools=STANDARD_TOOLS,          # 5 tools
        kwargs=dict(size_class="5b", max_iterations=7, context_window=8192),
    ),
    "7b": dict(
        model="ollama:mistral:7b-instruct",
        tools=FULL_TOOLS,              # 7 tools incl. code exec + web search
        kwargs=dict(size_class="7b", max_iterations=10, context_window=8192),
    ),
}


def main() -> None:
    size = sys.argv[1] if len(sys.argv) > 1 else "7b"
    cfg = CONFIGS[size]

    agent = SmallModelAgent(
        cfg["model"],
        tools=cfg["tools"],
        # Confirm anything that writes; block nothing outright.
        policy=PermissionPolicy(confirm_at_or_above=Severity.WRITE),
        user="jacob",
        role="developer",
        **cfg["kwargs"],
    )

    result = agent.run("What is 17 * 23, and what is today's date?")

    print("\n--- structured result ---")
    for step in result.steps:
        print(f"[{step.iteration}] thought={step.thought!r} tool={step.tool} "
              f"obs={str(step.observation)[:80]!r}")
    print(f"status={result.status}")
    print(f"answer={result.final_answer}")
    print(f"prompt tokens (cumulative estimate)={result.total_prompt_tokens}")
    print(f"audit log: {result.audit_path}")


if __name__ == "__main__":
    main()
