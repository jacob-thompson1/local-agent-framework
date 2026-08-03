# Model Guide: choosing and configuring small models

## Size classes and what to expect

The framework auto-detects a size class from the model name (`mistral:7b` → `7b`) and applies a profile; override with `size_class=` if detection guesses wrong.

**3B (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B).** One narrow job per session. 3–4 tools maximum, 5 iterations, always one few-shot example, 4K context assumption. Expect one or two malformed-JSON retries per session — the corrective loop handles them, but they cost an LLM call each. Pass an explicit `tool_subset` rather than relying on automatic pruning; a 3B model shown the *wrong* three tools will happily misuse them.

**~5B class (Phi-3-mini 3.8B, Qwen2.5-4B/5B variants).** 4–6 tools, 7 iterations. Phi models especially benefit from capped output length — they narrate.

**7–8B (Mistral-7B-Instruct, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct).** The local-agent sweet spot: 6–8 tools, 10 iterations, reliable JSON, multi-step tool chains including error recovery (feed a SQL error back and they usually fix the query). If your hardware runs a 7B at usable speed, skip the smaller classes.

**Quantization.** Q4_K_M is the pragmatic default; Q3 and below measurably degrades JSON discipline, which this framework depends on. If you see parse-retry rates climb, suspect the quant before the prompt.

## Family quirks (programmatically available as `FAMILY_HINTS`)

**Mistral** follows JSON-only instructions well but appends trailing prose; the parser strips it. Keep temperature ≤ 0.3 for tool calling. **Llama 3.x** wraps JSON in markdown fences and narrates before acting; the extractor and corrective retry handle both. Responds well to capitalized constraint words. **Qwen2.5** is the strongest JSON emitter per parameter and tolerates the most tools per size class; specify the answer language if the task is ambiguous. **Phi** is verbose — cap `max_tokens` at the provider level; strong on code tools, weaker on long tool chains. **Gemma** deployments sometimes reject system-role messages: if tool calling fails consistently, set `merge_system_into_user=True` on the agent.

## Recommended starting configurations

```python
# 3B — single-purpose lookup/calculation agent
SmallModelAgent("ollama:qwen2.5:3b", tools=BASIC_TOOLS,
                size_class="3b", max_iterations=5, context_window=4096)

# 7B — general local agent
SmallModelAgent("ollama:mistral:7b-instruct", tools=FULL_TOOLS,
                size_class="7b", max_iterations=10, context_window=8192)

# Hybrid — local-first with logged cloud fallback
HybridChatModel(primary_spec="ollama:qwen2.5:7b",
                fallback_spec="anthropic:claude-haiku-4-5")
```

Set Ollama's context explicitly if you raise `context_window` here — Ollama defaults to 2048/4096 for many models regardless of what the model supports: `ChatOllama(model=..., num_ctx=8192)` or `model_kwargs={"num_ctx": 8192}`.

## Local small model vs. larger/cloud model

Choose local-small when: data must stay on the machine; call volume makes API costs or rate limits painful; tasks are narrow, tool-shaped, and verifiable (the tool result, not the model's prose, carries the truth); latency of a round trip matters; or you're air-gapped. Choose larger/cloud when: tasks need genuine multi-step planning or synthesis; errors are expensive and hard to verify; context requirements exceed 8K meaningfully; or tool count is irreducibly high. The honest middle path is hybrid mode — but treat every `model_fallback` audit event as a data-governance event, because that is the moment content leaves the host.

## Measuring instead of guessing

`my-agent analyze` (or `ToolRegistry.analyze()`) reports each tool's measured prompt cost, total fixed overhead as a fraction of the context window, and warnings when the configuration exceeds the profile. Watch two live numbers in the logs: context utilization warnings (75% / 90%) and parse-retry frequency. Rising parse retries mean the model is overloaded — cut tools before cutting the task.
