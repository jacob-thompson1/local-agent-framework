"""v0.2 features working together: events, planning, editing, checkpoints,
cost caps, and (optionally) the TUI.

Run against a real local model:  python 05_v02_features.py
"""

from my_agent_framework import (
    CostTracker, EditSession, EventBus, HybridChatModel, PermissionPolicy,
    Severity, SmallModelAgent, stdin_confirmer,
)
from my_agent_framework.tools import read_file, list_directory, write_file

WORKSPACE = "./sandbox"  # the only directory the agent may modify

# One bus; the audit log and anything else you attach see the same stream.
bus = EventBus()
bus.subscribe(lambda e, p: print(f"  [cost] ${p['total_cost_usd']:.4f}"),
              "cost_update")
bus.subscribe(lambda e, p: print(f"  [diff]\n{p['diff']}"), "diff_proposed")
bus.subscribe(lambda e, p: print(f"  [checkpoint] {p['checkpoint_id'][:8]}"),
              "checkpoint")

# Local-first with a cloud fallback, capped at 25 cents for the whole session.
llm = HybridChatModel(
    primary_spec="ollama:mistral:7b-instruct",
    fallback_spec="anthropic:claude-haiku-4-5",
)
tracker = CostTracker(spend_cap_usd=0.25)

edits = EditSession()  # inherits the agent's workspace confinement

agent = SmallModelAgent(
    llm,
    tools=[read_file, list_directory, write_file],
    edit_session=edits,             # adds propose_edit / apply_edit
    workspace_root=WORKSPACE,        # confines writes + enables checkpoints
    plan_first=True,                 # plan -> approve -> act (7B+ advised)
    cost_tracker=tracker,
    bus=bus,
    policy=PermissionPolicy(
        confirm_at_or_above=Severity.WRITE,  # plan + every write needs a yes
        confirmer=stdin_confirmer,
    ),
    user="jacob", role="analyst",
)

if __name__ == "__main__":
    import os
    os.makedirs(WORKSPACE, exist_ok=True)
    result = agent.run(
        "Read every .txt file in the workspace and fix any line that "
        "says 'TODO' by replacing it with 'DONE'."
    )
    print(result.status, "->", result.final_answer)
    print("spend:", tracker.summary())
    # Something went wrong? Roll the workspace back:
    # agent.checkpointer.restore(agent.checkpointer.checkpoints[0].checkpoint_id,
    #                            mode="files")

    # Prefer watching it live? (pip install my-agent-framework[tui])
    # from my_agent_framework.tui import run_tui
    # from my_agent_framework.events import QueueConfirmer
    # run_tui(lambda bus, confirmer: SmallModelAgent(..., bus=bus,
    #         policy=PermissionPolicy(confirmer=confirmer)), task="...")
