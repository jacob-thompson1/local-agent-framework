"""Terminal UI for watching and approving agent runs (optional extra).

Install with ``pip install my-agent-framework[tui]``. Importing this module
is free; ``textual`` loads only when :func:`run_tui` is called -- the
zero-cost-at-import rule holds.

Architecture: the TUI is **just another event-bus subscriber**. It owns no
agent internals -- it renders the same stream the audit log records (plus the
ephemeral ``approval_request`` / ``cost_update`` / ``diff_proposed`` events)
and answers approvals through the ``respond`` callable that
:class:`~my_agent_framework.events.QueueConfirmer` puts in the payload. Any
other frontend (web, notebook, Slack bot) can subscribe to the identical
events and needs nothing further from the framework.

Because a Textual app owns the terminal, **never** pair the TUI with
``stdin_confirmer`` -- build your policy with a ``QueueConfirmer`` on the same
bus (``make_tui_agent`` does this for you)::

    from my_agent_framework.tui import run_tui
    result = run_tui(agent_factory=lambda bus, confirmer: SmallModelAgent(
        "ollama:mistral:7b-instruct", tools=FULL_TOOLS, bus=bus,
        policy=PermissionPolicy(confirmer=confirmer),
    ), task="Summarize sales.csv")
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from .events import EventBus, QueueConfirmer

__all__ = ["run_tui"]


def run_tui(
    agent_factory: Callable[[EventBus, QueueConfirmer], Any],
    task: str,
    approval_timeout_s: Optional[float] = 300.0,
) -> Any:
    """Run *task* with a live TUI. Returns the AgentResult.

    ``agent_factory(bus, confirmer)`` must build the agent using the provided
    bus and a policy whose confirmer is the provided ``QueueConfirmer``.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, RichLog, Static
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The TUI requires the 'textual' package: "
            "pip install my-agent-framework[tui]"
        ) from exc

    bus = EventBus()
    confirmer = QueueConfirmer(bus, timeout_s=approval_timeout_s)
    agent = agent_factory(bus, confirmer)

    class AgentTUI(App):
        CSS = """
        #plan, #cost { height: auto; min-height: 3; border: solid $accent;
                       padding: 0 1; }
        #diff { height: auto; max-height: 12; border: solid $warning;
                padding: 0 1; display: none; }
        #approval { height: auto; border: heavy $error; padding: 1;
                    display: none; }
        #log { border: solid $primary; }
        Button { margin-right: 2; }
        """
        BINDINGS = [("q", "quit", "Quit")]

        def __init__(self) -> None:
            super().__init__()
            self._respond: Optional[Callable[[bool], None]] = None
            self._result: Any = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("Plan: (none)", id="plan")
            yield Static("Cost: $0.000000 (0 calls)", id="cost")
            yield Static("", id="diff")
            with Vertical(id="approval"):
                yield Static("", id="approval_text")
                with Horizontal():
                    yield Button("Approve", id="approve", variant="success")
                    yield Button("Reject", id="reject", variant="error")
            yield RichLog(id="log", wrap=True, markup=False)
            yield Footer()

        # -- bus -> UI (thread-safe via call_from_thread) ------------------

        def on_mount(self) -> None:
            bus.subscribe(self._on_event)
            threading.Thread(target=self._run_agent, daemon=True).start()

        def _run_agent(self) -> None:
            try:
                self._result = agent.run(task)
            except Exception as exc:  # surfaced in the log, not swallowed
                self.call_from_thread(
                    self.query_one("#log", RichLog).write,
                    f"[agent crashed] {type(exc).__name__}: {exc}",
                )
            finally:
                self.call_from_thread(self._finish)

        def _finish(self) -> None:
            log = self.query_one("#log", RichLog)
            if self._result is not None:
                log.write(f"== {self._result.status.upper()} ==")
                log.write(str(self._result.final_answer))
            log.write("Press q to quit.")

        def _on_event(self, etype: str, payload: dict) -> None:
            self.call_from_thread(self._render_event, etype, payload)

        def _render_event(self, etype: str, payload: dict) -> None:
            log = self.query_one("#log", RichLog)
            if etype == "plan":
                steps = payload.get("steps") or []
                self.query_one("#plan", Static).update(
                    "Plan: " + ("; ".join(steps) if steps else "(none)")
                )
            elif etype == "cost_update":
                self.query_one("#cost", Static).update(
                    f"Cost: ${payload['total_cost_usd']:.6f} "
                    f"({payload['calls']} calls, last: {payload['provider']})"
                )
            elif etype == "diff_proposed":
                diff = self.query_one("#diff", Static)
                diff.update(
                    f"Proposed edit {payload['edit_id']} -> "
                    f"{payload['path']}\n{payload['diff']}"
                )
                diff.styles.display = "block"
            elif etype == "approval_request":
                self._respond = payload["respond"]
                self.query_one("#approval_text", Static).update(
                    f"APPROVE? {payload['tool']} [{payload['severity']}] "
                    f"args={payload['args']}\nreason: {payload['thought']}"
                )
                self.query_one("#approval").styles.display = "block"
            elif etype == "decision":
                log.write(
                    f"[{payload.get('iteration', '?')}] "
                    f"{payload.get('thought', '')} -> "
                    f"{payload.get('tool') or 'FINAL'}"
                )
            elif etype == "tool_result":
                ok = "ok" if payload.get("success") else "ERROR"
                log.write(f"    {payload.get('tool')}: {ok} "
                          f"{str(payload.get('result'))[:200]}")
            elif etype == "checkpoint":
                log.write(f"    checkpoint {payload['checkpoint_id'][:8]} "
                          f"({payload['backend']})")
            elif etype in ("error", "model_fallback", "compaction"):
                log.write(f"    [{etype}] {payload}")

        def on_button_pressed(self, event: "Button.Pressed") -> None:
            if self._respond is None:
                return
            self._respond(event.button.id == "approve")
            self._respond = None
            self.query_one("#approval").styles.display = "none"

    app = AgentTUI()
    app.run()
    return app._result
