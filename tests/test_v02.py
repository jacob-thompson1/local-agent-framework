"""Tests for v0.2 features: events, editing, checkpointing, planning,
compaction, cost tracking, workspace confinement."""

from __future__ import annotations

import pytest

from my_agent_framework import (
    CostTracker, EditSession, EventBus, PermissionPolicy, QueueConfirmer,
    Severity, SmallModelAgent, SpendCapExceeded, WorkspaceCheckpointer,
    flexible_find,
)
from my_agent_framework.memory import ConversationMemory
from my_agent_framework.tokens import TokenCounter


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "fake:test-7b"

    def invoke(self, messages, **kw):
        class M:
            def __init__(s, c):
                s.content = c
        return M(self.responses.pop(0))


def make_agent(responses, tmp_path, **kw):
    kw.setdefault("policy", PermissionPolicy(confirm_at_or_above=None))
    kw.setdefault("audit_root", tmp_path / "audit")
    return SmallModelAgent(FakeLLM(responses), **kw)


# -- events -----------------------------------------------------------------

def test_bus_mirrors_every_audit_event(tmp_path):
    seen = []
    bus = EventBus()
    bus.subscribe(lambda e, p: seen.append(e))
    agent = make_agent(
        ['{"thought": "done", "final": "42"}'], tmp_path, bus=bus,
    )
    agent.run("what is 6*7")
    assert "session_start" in seen and "llm_call" in seen
    assert "decision" in seen and "session_end" in seen


def test_bus_subscriber_exception_isolated(tmp_path):
    bus = EventBus()
    bus.subscribe(lambda e, p: 1 / 0)
    good = []
    bus.subscribe(lambda e, p: good.append(e))
    agent = make_agent(['{"thought": "x", "final": "ok"}'], tmp_path, bus=bus)
    result = agent.run("t")
    assert result.status == "success" and good


def test_queue_confirmer_approve_and_timeout():
    bus = EventBus()

    def approver(etype, payload):
        payload["respond"](True)

    bus.subscribe(approver, "approval_request")
    c = QueueConfirmer(bus, timeout_s=5)
    assert c("write_file", {}, Severity.WRITE, "test") is True
    # no subscriber answering -> fail closed
    c2 = QueueConfirmer(EventBus(), timeout_s=0.05)
    assert c2("write_file", {}, Severity.WRITE, "test") is False


# -- editing ----------------------------------------------------------------

def test_flexible_find_modes():
    hay = "line one\n    line two\nline three\n"
    m = flexible_find(hay, "line two")
    assert m.kind == "exact"
    m = flexible_find(hay, "line  two")  # whitespace-mangled
    assert m.kind == "whitespace_normalized"
    with pytest.raises(ValueError, match="not found"):
        flexible_find(hay, "line four")
    with pytest.raises(ValueError, match="matches 2"):
        flexible_find("a\nx\na\n", "a")


def test_edit_propose_apply_roundtrip(tmp_path):
    target = tmp_path / "code.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    session = EditSession(workspace_root=str(tmp_path))
    propose, apply_ = session.tools()
    out = propose("code.py" if False else str(target),
                  search="    return 1", replace="    return 2")
    assert "ready" in out
    edit_id = next(iter(session.pending))
    out = apply_(edit_id)
    assert "Applied" in out
    assert "return 2" in target.read_text()
    # stale apply: propose then change file underneath
    propose(str(target), search="    return 2", replace="    return 3")
    edit_id = next(iter(session.pending))
    target.write_text("completely different\n")
    assert "changed since" in apply_(edit_id)


def test_edit_workspace_confinement(tmp_path):
    inside = tmp_path / "ws"
    inside.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    session = EditSession(workspace_root=str(inside))
    propose, _ = session.tools()
    with pytest.raises(ValueError, match="outside the workspace"):
        session._resolve(str(outside))


def test_agent_run_with_edit_tools(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    f = ws / "a.txt"
    f.write_text("hello world\n")
    session = EditSession()
    agent = make_agent(
        [
            '{"thought": "propose", "tool": "propose_edit", "args": '
            f'{{"path": "{f}", "search": "hello world", "replace": "hi"}}}}',
            None,  # placeholder replaced below
            '{"thought": "done", "final": "edited"}',
        ],
        tmp_path, edit_session=session, workspace_root=str(ws),
    )

    # second response needs the runtime edit_id; patch the fake mid-flight
    real_invoke = agent._llm_input.invoke

    def invoke(messages, **kw):
        if agent._llm_input.responses[0] is None:
            edit_id = next(iter(session.pending))
            agent._llm_input.responses[0] = (
                '{"thought": "apply", "tool": "apply_edit", '
                f'"args": {{"edit_id": "{edit_id}"}}}}'
            )
        return real_invoke(messages, **kw)

    agent._llm_input.invoke = invoke
    result = agent.run("change hello to hi")
    assert result.status == "success"
    assert f.read_text().startswith("hi")


# -- checkpointing ----------------------------------------------------------

def test_checkpoint_restore_files_and_task(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("v1\n")
    cpt = WorkspaceCheckpointer(str(ws), shadow_root=str(tmp_path / "shadow"))
    mem = ConversationMemory(max_tokens=5000)
    mem.add("user", "task", pinned=True)
    cp = cpt.checkpoint("t0", memory_turns=len(mem.turns))
    (ws / "a.txt").write_text("v2\n")
    (ws / "junk.txt").write_text("x\n")
    mem.add("assistant", "did stuff")
    out = cpt.restore(cp.checkpoint_id, mode="both", memory=mem)
    assert (ws / "a.txt").read_text() == "v1\n"
    assert not (ws / "junk.txt").exists()
    assert len(mem.turns) == 1 and out["turns_removed"] == 1


def test_agent_auto_checkpoints_after_write(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    from my_agent_framework.tools import write_file
    events = []
    bus = EventBus()
    bus.subscribe(lambda e, p: events.append(e), "checkpoint")
    agent = make_agent(
        [
            '{"thought": "write", "tool": "write_file", "args": '
            f'{{"path": "{ws / "out.txt"}", "content": "data"}}}}',
            '{"thought": "done", "final": "written"}',
        ],
        tmp_path, tools=[write_file], workspace_root=str(ws), bus=bus,
        checkpointer=WorkspaceCheckpointer(
            str(ws), shadow_root=str(tmp_path / "shadow")),
    )
    result = agent.run("write a file")
    assert result.status == "success"
    assert events == ["checkpoint"]
    assert len(agent.checkpointer.checkpoints) == 1


def test_agent_workspace_confinement_blocks(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    from my_agent_framework.tools import write_file
    agent = make_agent(
        [
            '{"thought": "escape", "tool": "write_file", "args": '
            f'{{"path": "{tmp_path / "outside.txt"}", "content": "x"}}}}',
            '{"thought": "give up", "final": "blocked"}',
        ],
        tmp_path, tools=[write_file], workspace_root=str(ws),
    )
    result = agent.run("write outside")
    assert result.status == "success"
    assert not (tmp_path / "outside.txt").exists()
    assert result.steps[0].permission_outcome == "blocked"


# -- planning ---------------------------------------------------------------

def test_plan_first_pins_plan_and_runs(tmp_path):
    agent = make_agent(
        [
            '{"plan": ["compute the product", "answer"]}',
            '{"thought": "done", "final": "42"}',
        ],
        tmp_path, plan_first=True,
    )
    events = []
    agent.bus.subscribe(lambda e, p: events.append((e, p)), "plan")
    result = agent.run("6*7")
    assert result.status == "success"
    assert events and events[0][1]["steps"] == ["compute the product", "answer"]


def test_plan_rejected_aborts(tmp_path):
    agent = make_agent(
        ['{"plan": ["nuke everything"]}'],
        tmp_path, plan_first=True,
        policy=PermissionPolicy(
            confirm_at_or_above=Severity.WRITE,
            confirmer=lambda *a: False,
        ),
    )
    result = agent.run("dangerous thing")
    assert result.status == "plan_rejected"
    assert agent._llm_input.responses == []  # no act-phase calls made


def test_unparseable_plan_proceeds_planless(tmp_path):
    agent = make_agent(
        ["I refuse to emit JSON", '{"thought": "d", "final": "ok"}'],
        tmp_path, plan_first=True,
    )
    result = agent.run("t")
    assert result.status == "success"


# -- compaction -------------------------------------------------------------

def test_proactive_compaction_with_hysteresis():
    mem = ConversationMemory(
        max_tokens=100, counter=TokenCounter(),
        summarizer=lambda text: "summary of old turns",
    )
    for i in range(30):
        mem.add("tool", f"observation {i} " + "x" * 40)
    assert mem.compaction_history, "compaction never triggered"
    entry = mem.compaction_history[0]
    assert entry["trigger"] == "proactive_75pct"
    assert entry["tokens_after"] <= int(100 * 0.75)  # trimmed past trigger
    assert "summary of old turns" in mem.summary
    # reactive without summarizer
    mem2 = ConversationMemory(max_tokens=100, counter=TokenCounter())
    for i in range(30):
        mem2.add("tool", f"obs {i} " + "y" * 40)
    assert mem2.compaction_history[0]["trigger"] == "hard_budget"


def test_compaction_audited(tmp_path):
    long_obs = "word " * 300
    agent = make_agent(
        [
            '{"thought": "look", "tool": "get_current_time", "args": {}}',
            '{"thought": "done", "final": "ok"}',
        ],
        tmp_path, memory_tokens=120,
    )
    from my_agent_framework.tools import get_current_time
    agent.registry.register(get_current_time)
    seen = []
    agent.bus.subscribe(lambda e, p: seen.append(p), "compaction")
    agent.memory.summarizer = lambda t: "compressed"
    result = agent.run("task " + long_obs)
    assert result.status == "success"


# -- cost tracking ----------------------------------------------------------

def test_cost_tracker_records_and_caps():
    t = CostTracker(spend_cap_usd=0.01)
    p = t.record("ollama:qwen2.5:7b", 5000, 500)
    assert p["call_cost_usd"] == 0.0 and t.total_prompt_tokens == 5000
    t.record("anthropic:claude-haiku-4-5", 1_000_000, 100_000)
    assert t.total_cost_usd > 0.01
    with pytest.raises(SpendCapExceeded):
        t.check_cap("anthropic:claude-haiku-4-5", 1000)
    t.check_cap("ollama:qwen2.5:7b", 10**9)  # local never capped


def test_cost_events_on_bus(tmp_path):
    tracker = CostTracker()
    seen = []
    agent = make_agent(
        ['{"thought": "d", "final": "ok"}'], tmp_path, cost_tracker=tracker,
    )
    agent.bus.subscribe(lambda e, p: seen.append(p), "cost_update")
    tracker.record("openai:gpt-4o-mini", 1000, 100)
    assert seen and seen[0]["total_cost_usd"] > 0


def test_pricing_lookup_longest_match():
    from my_agent_framework.tokens import lookup_price
    assert lookup_price("openai:gpt-4o-mini") == (0.15, 0.60)
    assert lookup_price("openai:gpt-4o") == (2.50, 10.00)
    assert lookup_price("ollama:mystery-model") is None


# -- tui module -------------------------------------------------------------

def test_tui_imports_without_textual():
    import my_agent_framework.tui as tui
    assert callable(tui.run_tui)
