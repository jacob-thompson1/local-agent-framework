"""In-process event bus: one stream for audit, UIs, and custom subscribers.

Design: the framework has exactly **one emission point** for everything that
gets recorded -- :meth:`AuditLogger.event`. When an :class:`EventBus` is
attached to the logger, every audit record is mirrored onto the bus, so a UI
(the Textual TUI, your own dashboard, a test) sees precisely the stream the
compliance log sees and the two can never disagree.

A few *ephemeral* events are bus-only because they are interaction, not
record: ``approval_request`` (carries a live ``respond`` callable) and
``cost_update`` ticks. Their audit-relevant outcomes (the approval decision,
the session token totals) still land in the audit log through the normal
events.

Event names seen on the bus (payload = the audit record dict unless noted):

* audit-mirrored: ``session_start``, ``llm_call``, ``decision``, ``approval``,
  ``tool_result``, ``error``, ``model_fallback``, ``session_end``, ``plan``,
  ``compaction``, ``checkpoint``, ``restore``
* bus-only: ``approval_request`` (payload has ``tool``, ``args``, ``severity``,
  ``thought``, ``respond(bool)``), ``cost_update`` (payload has ``provider``,
  ``call_cost``, ``total_cost``, ``prompt_tokens``, ``completion_tokens``),
  ``diff_proposed`` (payload has ``edit_id``, ``path``, ``diff``)

Subscriber exceptions are caught and logged, never propagated into the agent
loop. No threads are created here; ``emit`` is synchronous. Zero imports
beyond the standard library.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger("my_agent_framework.events")

Subscriber = Callable[[str, dict], None]


class EventBus:
    """Synchronous publish/subscribe hub.

    ``subscribe(fn)`` receives every event; ``subscribe(fn, "decision")``
    receives one type. Returns an unsubscribe callable.
    """

    def __init__(self) -> None:
        self._subs: list[tuple[Optional[str], Subscriber]] = []
        self._lock = threading.Lock()

    def subscribe(
        self, fn: Subscriber, event_type: Optional[str] = None
    ) -> Callable[[], None]:
        entry = (event_type, fn)
        with self._lock:
            self._subs.append(entry)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)

        return unsubscribe

    def emit(self, event_type: str, payload: Optional[dict] = None) -> None:
        payload = payload or {}
        with self._lock:
            subs = list(self._subs)
        for wanted, fn in subs:
            if wanted is None or wanted == event_type:
                try:
                    fn(event_type, payload)
                except Exception:
                    logger.exception(
                        "Event subscriber %r raised on %r; continuing.", fn, event_type
                    )


class QueueConfirmer:
    """A :class:`~my_agent_framework.safety.PermissionPolicy` confirmer that
    routes approval through the event bus instead of stdin.

    Emits ``approval_request`` with a ``respond(bool)`` callable in the
    payload, then blocks (with optional timeout) until some subscriber -- a
    TUI approval dialog, a web handler, a test -- calls it. Timeout or no
    subscriber answering means **rejected** (fail closed).

    Usage::

        bus = EventBus()
        policy = PermissionPolicy(confirmer=QueueConfirmer(bus, timeout_s=120))
    """

    def __init__(self, bus: EventBus, timeout_s: Optional[float] = 300.0) -> None:
        self.bus = bus
        self.timeout_s = timeout_s

    def __call__(self, name: str, args: dict, severity: Any, thought: str) -> bool:
        answer: "queue.Queue[bool]" = queue.Queue(maxsize=1)

        def respond(approved: bool) -> None:
            try:
                answer.put_nowait(bool(approved))
            except queue.Full:  # already answered; ignore duplicates
                pass

        self.bus.emit(
            "approval_request",
            {
                "tool": name,
                "args": args,
                "severity": getattr(severity, "value", str(severity)),
                "thought": thought,
                "respond": respond,
            },
        )
        try:
            return answer.get(timeout=self.timeout_s)
        except queue.Empty:
            logger.warning(
                "Approval request for %r timed out after %ss; rejecting "
                "(fail closed).", name, self.timeout_s,
            )
            return False
