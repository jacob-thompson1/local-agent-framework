"""Plugging in different LLMs: OpenAI, Anthropic, custom endpoints, hybrid.

The agent accepts (a) a provider string, (b) any LangChain BaseChatModel
instance, or (c) a HybridChatModel. Credentials come from your OS credential
manager (via `my-agent config set anthropic-api-key ...`) or standard env vars
(ANTHROPIC_API_KEY, OPENAI_API_KEY) read by the provider SDKs. Nothing is
contacted until .run() actually invokes the model.
"""

from __future__ import annotations

from my_agent_framework import HybridChatModel, SmallModelAgent
from my_agent_framework.tools import BASIC_TOOLS

# --- (a) Provider strings ---------------------------------------------------
openai_agent = SmallModelAgent("openai:gpt-4o-mini", tools=BASIC_TOOLS)
anthropic_agent = SmallModelAgent("anthropic:claude-haiku-4-5", tools=BASIC_TOOLS)

# Custom OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server):
custom_agent = SmallModelAgent(
    "custom:my-finetune-7b",
    tools=BASIC_TOOLS,
    model_kwargs={"base_url": "http://localhost:8000/v1"},
)

# --- (b) A pre-built LangChain model object ---------------------------------
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="qwen2.5:7b", temperature=0.1, num_ctx=8192)
# agent = SmallModelAgent(llm, tools=BASIC_TOOLS, size_class="7b")

# --- (c) Hybrid: local-first, cloud fallback --------------------------------
# The fallback is only constructed and called if the local model errors
# (Ollama down, OOM, etc.). Every fallback is logged in the audit trail as a
# `model_fallback` event, since data leaves the machine at that point.
hybrid = HybridChatModel(
    primary_spec="ollama:mistral:7b-instruct",
    fallback_spec="anthropic:claude-haiku-4-5",
)
hybrid_agent = SmallModelAgent(hybrid, tools=BASIC_TOOLS, size_class="7b")

if __name__ == "__main__":
    result = hybrid_agent.run("What is 91 / 7?")
    print(result.final_answer)
    print("served by:", result.model_used)
