"""Cross-platform test suite (runs identically on Windows/macOS/Linux).

Uses a scripted fake LLM -- no model, no network, no GPU required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from my_agent_framework import (
    AuditExporter,
    PermissionPolicy,
    Settings,
    Severity,
    SmallModelAgent,
    ToolRegistry,
    TokenCounter,
    detect_size_class,
    read_only_policy,
    tool,
)
from my_agent_framework.agent import extract_json_object


class FakeLLM:
    """Returns scripted responses in order; mimics BaseChatModel.invoke."""

    model = "fake:test-7b"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        class Msg:
            def __init__(self, content: str) -> None:
                self.content = content

        self.calls += 1
        return Msg(self.responses.pop(0) if self.responses else '{"final": "done"}')


@tool("Add two numbers.", keywords=["math", "add"])
def add(a: float, b: float) -> float:
    return a + b


@tool("Delete everything.", severity=Severity.DESTRUCTIVE, keywords=["delete"])
def nuke(target: str) -> str:
    return f"deleted {target}"


@pytest.fixture()
def audit_root(tmp_path: Path) -> Path:
    return tmp_path / "audit"


def make_agent(responses, tools, audit_root, **kw):
    kw.setdefault("policy", PermissionPolicy(confirm_at_or_above=None))
    return SmallModelAgent(
        FakeLLM(responses), tools=tools, audit_root=audit_root,
        size_class="7b", **kw,
    )


# -- end-to-end loop ---------------------------------------------------------

def test_agent_completes_tool_task(audit_root):
    agent = make_agent(
        [
            '{"thought": "use add", "tool": "add", "args": {"a": 2, "b": 3}}',
            '{"thought": "got it", "final": "The answer is 5."}',
        ],
        [add], audit_root,
    )
    result = agent.run("What is 2 + 3?")
    assert result.status == "success"
    assert result.final_answer == "The answer is 5."
    assert result.steps[0].tool == "add"
    assert result.steps[0].observation == "5"
    assert result.total_prompt_tokens > 0


def test_malformed_json_recovers(audit_root):
    agent = make_agent(
        [
            "Sure! Let me think about this...",  # unparseable
            '```json\n{"thought": "ok", "final": "42"}\n```',  # fenced
        ],
        [add], audit_root,
    )
    result = agent.run("Meaning of life?")
    assert result.status == "success"
    assert result.final_answer == "42"


def test_unknown_tool_reported_back(audit_root):
    agent = make_agent(
        [
            '{"thought": "hm", "tool": "missing_tool", "args": {}}',
            '{"thought": "fallback", "final": "done without it"}',
        ],
        [add], audit_root,
    )
    result = agent.run("Do something")
    assert result.status == "success"
    assert "not an available tool" in result.steps[0].observation


def test_max_iterations(audit_root):
    agent = make_agent(
        ['{"thought": "loop", "tool": "add", "args": {"a": 1, "b": 1}}'] * 10,
        [add], audit_root, max_iterations=3,
    )
    result = agent.run("Loop forever")
    assert result.status == "max_iterations"
    assert result.iterations == 3


def test_tool_error_is_survivable(audit_root):
    @tool("Always fails.")
    def broken() -> str:
        raise RuntimeError("boom")

    agent = make_agent(
        [
            '{"thought": "try", "tool": "broken", "args": {}}',
            '{"thought": "it failed", "final": "tool was broken"}',
        ],
        [broken], audit_root,
    )
    result = agent.run("Use the broken tool")
    assert result.status == "success"
    assert "boom" in result.steps[0].observation


# -- safety ------------------------------------------------------------------

def test_read_only_policy_blocks_destructive(audit_root):
    agent = SmallModelAgent(
        FakeLLM([
            '{"thought": "destroy", "tool": "nuke", "args": {"target": "x"}}',
            '{"thought": "blocked", "final": "could not delete"}',
        ]),
        tools=[nuke], audit_root=audit_root, size_class="7b",
        policy=read_only_policy(),
    )
    result = agent.run("Delete x")
    assert result.steps[0].permission_outcome == "blocked"
    assert "deleted" not in (result.steps[0].observation or "")


def test_confirmation_rejection(audit_root):
    policy = PermissionPolicy(
        confirm_at_or_above=Severity.DESTRUCTIVE,
        confirmer=lambda name, args, sev, reason: False,
    )
    agent = SmallModelAgent(
        FakeLLM([
            '{"thought": "destroy", "tool": "nuke", "args": {"target": "x"}}',
            '{"final": "user said no"}',
        ]),
        tools=[nuke], audit_root=audit_root, size_class="7b", policy=policy,
    )
    result = agent.run("Delete x")
    assert result.steps[0].permission_outcome == "rejected"


def test_dry_run_executes_nothing(audit_root):
    executed = []

    @tool("Record execution.", severity=Severity.WRITE)
    def side_effect(x: str) -> str:
        executed.append(x)
        return "did it"

    policy = PermissionPolicy(dry_run=True)
    agent = SmallModelAgent(
        FakeLLM([
            '{"thought": "do", "tool": "side_effect", "args": {"x": "hello"}}',
            '{"final": "planned"}',
        ]),
        tools=[side_effect], audit_root=audit_root, size_class="7b",
        policy=policy,
    )
    result = agent.run("Do the thing")
    assert executed == []
    assert result.status == "dry_run"
    assert "DRY RUN" in result.final_answer


def test_tool_timeout(audit_root):
    import time as _time

    @tool("Sleeps too long.", timeout_s=0.2)
    def sleepy() -> str:
        _time.sleep(2)
        return "never"

    agent = make_agent(
        [
            '{"thought": "wait", "tool": "sleepy", "args": {}}',
            '{"final": "it timed out"}',
        ],
        [sleepy], audit_root,
    )
    result = agent.run("Wait")
    assert "timeout" in result.steps[0].observation.lower()


# -- audit -------------------------------------------------------------------

def test_audit_trail_complete_and_exportable(audit_root):
    agent = make_agent(
        [
            '{"thought": "use add", "tool": "add", "args": {"a": 1, "b": 2}}',
            '{"final": "3"}',
        ],
        [add], audit_root, user="jacob", role="analyst",
    )
    result = agent.run("1+2?")
    exporter = AuditExporter(root=audit_root)
    records = list(exporter.iter_records(session_id=result.session_id))
    events = [r["event"] for r in records]
    for expected in ("session_start", "llm_call", "decision",
                     "approval", "tool_result", "session_end"):
        assert expected in events
    start = next(r for r in records if r["event"] == "session_start")
    assert start["user"] == "jacob" and start["role"] == "analyst"
    assert start["model"] == "fake:test-7b"
    assert start["tools_available"][0]["name"] == "add"
    # exports
    parsed = json.loads(exporter.export("json", session_id=result.session_id))
    assert len(parsed) == len(records)
    csv_text = exporter.export("csv", session_id=result.session_id)
    assert "session_start" in csv_text
    summary = exporter.summary()
    assert summary["sessions"] == 1
    assert summary["tool_usage"]["add"] == 1


def test_redaction(audit_root, tmp_path):
    agent = make_agent(
        ['{"thought": "note SSN 123-45-6789", "final": "ok"}'],
        [add], audit_root,
    )
    result = agent.run("Handle claim for someone@example.com")
    from my_agent_framework import DEFAULT_REDACTION_PATTERNS
    exporter = AuditExporter(
        root=audit_root, redact_patterns=DEFAULT_REDACTION_PATTERNS)
    text = exporter.export("json", session_id=result.session_id)
    assert "123-45-6789" not in text
    assert "someone@example.com" not in text
    assert "[REDACTED:ssn]" in text


# -- registry / tokens -------------------------------------------------------

def test_pruning_prefers_relevant_tools():
    registry = ToolRegistry()
    registry.register_all([add, nuke])

    @tool("Fetch the weather forecast.", keywords=["weather", "forecast"])
    def weather(city: str) -> str:
        return "sunny"

    registry.register(weather)
    selected = registry.select_for_task("What is the weather in Omaha?", max_tools=1)
    assert [t.name for t in selected] == ["weather"]


def test_analyze_warns_on_overload():
    registry = ToolRegistry()
    registry.register_all([add, nuke])
    report = registry.analyze("3b", context_window=4096)
    assert report["tool_count"] == 2
    assert report["total_tool_tokens"] > 0
    heavy = registry.analyze("3b", context_window=512, system_prompt_tokens=400)
    assert not heavy["ok"]


def test_token_counter_and_size_detection():
    counter = TokenCounter()
    assert counter.count("hello world, this is a test") >= 5
    assert detect_size_class("mistral:7b-instruct") == "7b"
    assert detect_size_class("qwen2.5:3b") == "3b"
    assert detect_size_class("phi3:mini-4b") == "5b"
    assert detect_size_class("gpt-4o-mini") == "large"


def test_extract_json_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Sure!\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    assert extract_json_object('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    assert extract_json_object("no json here") is None


# -- config ------------------------------------------------------------------

def test_settings_roundtrip_and_env_override(tmp_path, monkeypatch):
    settings = Settings(path=tmp_path / "config.json")
    settings.set("model", "ollama:qwen2.5:3b")
    settings.set("hybrid-mode", "true")
    reloaded = Settings(path=tmp_path / "config.json")
    assert reloaded.get("model") == "ollama:qwen2.5:3b"
    assert reloaded.get("hybrid-mode") is True
    monkeypatch.setenv("MY_AGENT_MODEL", "ollama:mistral:7b")
    assert reloaded.get("model") == "ollama:mistral:7b"
    reloaded.reset()
    assert not (tmp_path / "config.json").exists()


# -- zero network at import --------------------------------------------------

def test_import_makes_no_network_calls():
    """Import the package in a subprocess with sockets disabled."""
    code = (
        "import socket\n"
        "def _blocked(*a, **k): raise AssertionError('network at import!')\n"
        "socket.socket.connect = _blocked\n"
        "socket.create_connection = _blocked\n"
        "socket.getaddrinfo = _blocked\n"
        "import my_agent_framework\n"
        "import my_agent_framework.tools\n"
        "from my_agent_framework import SmallModelAgent\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert "CLEAN" in proc.stdout
