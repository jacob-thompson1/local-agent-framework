# my-agent-framework

A production-oriented agent framework built on LangChain, designed **small-model-first**: every architectural decision assumes your LLM is a 3B–7B parameter model running on local hardware with a 4K–8K context window, and that every token in the prompt has to earn its place.

Any LangChain chat model plugs in unchanged — local Ollama, OpenAI, Anthropic, or any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server) — but the defaults, prompts, and guardrails are tuned for the small end.

## Why this exists

Today, the wrapper is everything. Stock agent frameworks and closed commercial wrappers assume frontier models: verbose system prompts, native function calling, twenty tools in context, unbounded history. Point them at a 3B model and you get wrong-tool selection, malformed JSON, and context overflow. Access to many of the best wrappers can be difficult to obtain or unsafe to deploy in corporate environments due to data residency and compliance concerns.

Frameworks like LangGraph and LangChain can work as alternatives in these settings, but they require significant architectural complexity. Meanwhile, agentic platforms (n8n, Make, etc.) are often locked down in corporate IT policies, add unnecessary overhead, can be slow, and are better suited for prototyping than production deployment.

**This project bridges that gap.** It provides a lightweight agent framework that speeds up development while offering the full customization of building agents from scratch — suitable for organizations with domain-specific requirements and complex, custom skills. It's especially valuable for regulated industries (finance, healthcare, insurance) where compliance logging and data locality matter.

The framework automatically adjusts to your model size: instead of assuming frontier capabilities, it matches tool availability and architectural choices to the model's parameter count and context window. Empirically, the inflection points are around 7B, 20B, and 70B parameters, but these are configurable to fit your needs.

### Core design decisions

- **Strict single-JSON-object protocol** instead of native function calling, with corrective retries — works on every instruction-tuned model from 3B up.
- **Measured token accounting.** Every tool's prompt cost is measured at registration; the context budget is tracked every iteration and you're warned before you hit the wall.
- **Tool pruning by model size.** The registry holds your full inventory; each session gets only the tools relevant to the task and your model's capacity.
- **Approve-before-act safety** (Claude Code-style): severity levels, confirmation mode, read-only policies, dry-run.
- **Compliance-grade audit trail** for regulated environments: every decision, reasoning step, approval, error, and fallback logged per session, exportable as JSON/CSV with PII redaction.
- **Event bus** for live interaction: approval requests, cost updates, diff previews — audit log and UI never disagree.
- **Diff-based editing** via `propose_edit`/`apply_edit` using search/replace blocks (small models can't parse unified diffs). Proposing is READ_ONLY; applying is WRITE and gated by policy.
- **Workspace + checkpoints:** confine writes to one directory, auto-snapshot after edits, roll back files or conversation state.
- **Plan → approve → act:** optional one-step planning before tool execution, with confirmation gates.
- **Context-aware memory compaction** with hysteresis, and cost tracking with spend caps.
- **Zero external calls at import.** No telemetry, no version checks, no downloads. Provider SDKs load lazily; network activity happens only when needed.

### Corporate compliance note

**This framework is designed for organizations that want to deploy custom agents within their infrastructure and compliance boundaries.** It is **not** intended to help individuals circumvent IT policies, corporate security controls, or data governance requirements. Always follow your organization's policies regarding AI tools, data handling, and external services. If your organization restricts agentic software or third-party wrappers, consult your IT and legal teams before using any framework — including this one.

## Installation

```bash
pip install my-agent-framework[ollama]        # local models via Ollama
pip install my-agent-framework[all]           # everything (openai, anthropic, keyring, web search, tiktoken)
```

From this source tree: `pip install -e ".[ollama,dev]"`

Core dependencies are just `langchain-core` and `platformdirs` — pure Python, available on PyPI for Windows, macOS, and Linux. Extras: `[openai]`, `[anthropic]`, `[keyring]` (OS credential-manager storage for API keys), `[web]` (web search tool), `[tokenizer]` (tiktoken for exact counts instead of the heuristic).

**Hardware guidance.** 3B models (Qwen2.5-3B, Llama-3.2-3B) run comfortably in 4 GB of RAM/VRAM via Ollama; 7B models (Mistral-7B-Instruct, Qwen2.5-7B, Llama-3.1-8B) want 8 GB with Q4 quantization. Prefer instruction-tuned variants over base models, always.

## Sixty-second quickstart

```python
from my_agent_framework import SmallModelAgent, tool, PermissionPolicy, Severity

@tool("Add two numbers.", keywords=["math"])
def add(a: float, b: float) -> float:
    return a + b

agent = SmallModelAgent(
    "ollama:mistral:7b-instruct",                     # or any BaseChatModel
    tools=[add],
    policy=PermissionPolicy(confirm_at_or_above=Severity.WRITE),
)
result = agent.run("What is 41.5 + 0.5?")
print(result.final_answer)      # "42.0"
print(result.status)            # "success"
print(result.audit_path)        # per-session JSONL audit log
```

`result` is a structured `AgentResult`: status, final answer, per-step reasoning/tool/observation records, token totals, session ID, and the audit log path. See `examples/` for Ollama size configs, provider swapping, hybrid local→cloud fallback, a custom SQL tool, and the safety modes.

## Configure once, reuse forever

```bash
my-agent config set model ollama:mistral:7b-instruct
my-agent config set hybrid-mode true
my-agent config set fallback-model anthropic:claude-haiku-4-5
my-agent config set anthropic-api-key sk-ant-...   # -> OS credential manager, never plain text
my-agent config show
my-agent config reset
```

Settings live at the platform-standard config location (`~/.config/my-agent-framework/config.json` on Linux, `~/Library/Application Support/...` on macOS, `%APPDATA%\my-agent-framework\...` on Windows). API keys never touch the JSON file — they go through `keyring` (Windows Credential Manager / macOS Keychain / Secret Service), and the framework refuses to store them in plain text if keyring is missing. Every setting can be overridden by an environment variable (`MY_AGENT_MODEL`, `MY_AGENT_DRY_RUN`, ...) for automation.

Then:

```bash
my-agent run "What is 12 * 7?"
my-agent run "Summarize notes.txt" --tools read_file --dry-run
my-agent analyze                    # is my tool set too heavy for my model?
my-agent export-audit --session-id <id> --format json
my-agent export-audit --date-range 2026-01-01:2026-06-30 --redact -o q2_audit.json
my-agent audit-summary --date-range 2026-01-01:2026-06-30
```

In code, `agent_from_settings(tools=...)` builds an agent from the stored config.

## Token budgets and tool costs

Every tool definition's exact prompt cost is measured when you register it. The built-in tools run ~15–30 tokens each; the base system prompt is ~120 tokens (plus ~55 for the optional one-shot example). Rules of thumb, enforced as warnings by `my-agent analyze` and the `ToolRegistry.analyze()` API:

| Model size | Max tools | Comfortable | Context assumed | Iterations |
|-----------:|----------:|------------:|----------------:|-----------:|
| 3B         | 4         | 3           | 4,096           | 5          |
| ~5B        | 6         | 4           | 8,192           | 7          |
| 7–8B       | 8         | 6           | 8,192           | 10         |
| 13B+/cloud | 20        | 12          | 32,768          | 15         |

Keep tool descriptions under ~50 tokens; keep fixed overhead (system prompt + tools) under 35% of the context window. When the registry holds more tools than the budget allows, `select_for_task()` prunes by offline keyword relevance — no embeddings, no network — and logs exactly what was kept, dropped, and how many tokens were saved. For 3B models, prefer passing an explicit `tool_subset=[...]` per task over automatic pruning.

Context is tracked live: memory trims oldest-first within a token budget (with an optional summarizer callback), and you get one warning at 75% utilization and another at 90%.

## Safety model

Tools declare a severity: `READ_ONLY`, `WRITE`, or `DESTRUCTIVE`. A `PermissionPolicy` decides what happens:

```python
PermissionPolicy(confirm_at_or_above=Severity.WRITE)    # confirm writes & worse
read_only_policy()                                      # hard-block anything that isn't read-only
PermissionPolicy(dry_run=True)                          # execute nothing; report the plan
PermissionPolicy(always_confirm={"run_python"})         # per-tool overrides
```

The default confirmer prompts on stdin; supply your own callable for GUI/web apps. Blocks, approvals, and rejections are all fed back to the model (so it can route around a denial) and all written to the audit log.

## When to use a small local model vs. a larger one

Local 3B–7B models are the right call when data cannot leave the machine (claims data, PII, anything a regulator will ask about), when per-call cost or latency at volume matters, and when tasks are narrow and tool-shaped: lookups, calculations, structured queries, file operations, single-purpose pipelines. They are the wrong call for open-ended multi-step planning, subtle synthesis across long documents, or anything where a wrong-but-confident answer is expensive — that's what `hybrid-mode` is for: run local first, fall back to a cloud model on failure, with every fallback logged as a `model_fallback` audit event because that's the moment data leaves the host. Full guidance, including per-family behavior notes for Mistral/Llama/Qwen/Phi/Gemma, is in `docs/MODEL_GUIDE.md`.

## Docs

- `docs/MODEL_GUIDE.md` — model selection, family quirks, recommended configs per size
- `docs/COMPLIANCE_GUIDE.md` — what the audit log contains, exports, redaction, retention, demonstrating oversight to a regulator
- `docs/CROSS_PLATFORM.md` — per-OS setup, Ollama connectivity, path/encoding troubleshooting

## Project layout

```
my_agent_framework/
    agent.py        # SmallModelAgent: JSON ReAct loop, retries, timeouts
    registry.py     # @tool decorator, ToolRegistry, pruning, analyzer
    tokens.py       # TokenCounter, ContextBudget
    memory.py       # token-budgeted conversation memory
    safety.py       # Severity, PermissionPolicy, confirmers, dry-run
    audit.py        # AuditLogger (JSONL), AuditExporter (JSON/CSV, redaction)
    models.py       # profiles, family hints, lazy providers, HybridChatModel
    config.py       # persistent settings, keyring secrets, env overrides
    cli.py          # my-agent config|run|analyze|export-audit|audit-summary
    tools/          # built-in example tools with measured token costs
examples/           # runnable scripts (Ollama sizes, providers, SQL tool, safety)
tests/              # pytest suite incl. zero-network-at-import verification
.github/workflows/  # CI matrix: Windows + macOS + Linux × Python 3.10–3.12
```

## Development

```bash
pip install -e ".[dev]"
pytest              # 17 tests, no model or network required (scripted fake LLM)
ruff check .
```

MIT licensed.
