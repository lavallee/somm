## Purpose

This package lets LangChain, LangGraph, and Deep Agents applications use somm as a standard chat model. It preserves LangChain call sites while somm handles provider routing, telemetry, cost tracking, fallback, and model memory (`packages/somm-langchain/README.md:3`).

## How it works

`SommChatModel` subclasses LangChain’s `BaseChatModel`. During generation, it extracts and joins system messages, translates human, assistant, and tool messages into somm’s provider-neutral format, then calls `SommLLM.generate()` with workload, routing pins, sampling settings, and tools (`packages/somm-langchain/src/somm_langchain/chat_model.py:79`). System prompts are passed separately; assistant tool calls become `tool_use` blocks, while tool results become user-side `tool_result` blocks (`packages/somm-langchain/src/somm_langchain/chat_model.py:173`).

The returned somm result becomes a LangChain `AIMessage` inside a `ChatGeneration`, carrying tool calls, provider/model provenance, latency, cost, outcome, and token usage (`packages/somm-langchain/src/somm_langchain/chat_model.py:118`). `bind_tools()` first accepts LangChain-compatible tool definitions, normalizes them through OpenAI’s common schema, and unwraps them to somm’s neutral schema immediately before generation (`packages/somm-langchain/src/somm_langchain/chat_model.py:152`).

## Key surfaces

- `SommChatModel` — construct with a `SommLLM`, then use standard LangChain operations such as `invoke()` (`packages/somm-langchain/src/somm_langchain/chat_model.py:32`).
- `SommChatModel.bind_tools()` — returns a runnable with tool schemas and optional tool-choice policy bound to calls (`packages/somm-langchain/src/somm_langchain/chat_model.py:152`).
- `somm_langchain.SommChatModel` — the package’s sole public export (`packages/somm-langchain/src/somm_langchain/__init__.py:3`).

## Design decisions

- Failures raise by default so LangChain retry or circuit-breaker middleware can react; callers may instead request an empty message carrying failure metadata (`packages/somm-langchain/src/somm_langchain/chat_model.py:63`).
- Model and provider pins are optional, leaving routing to somm unless explicitly overridden (`packages/somm-langchain/src/somm_langchain/chat_model.py:57`).
- Text-only assistant turns collapse to plain strings for interoperability, while mixed text/tool turns retain structured blocks (`packages/somm-langchain/src/somm_langchain/chat_model.py:236`).
- Reasoning content is preserved across tool-calling turns because some thinking-model providers reject subsequent requests without it (`packages/somm-langchain/src/somm_langchain/chat_model.py:122`).
- Unknown LangChain message subclasses are forwarded as user text on a best-effort basis rather than rejected (`packages/somm-langchain/src/somm_langchain/chat_model.py:210`).
- The inspected implementation defines synchronous generation but no explicit streaming override, so package-specific streaming behavior is unclear.

## One-liner

`somm-langchain` is a thin bidirectional adapter that makes somm’s routed, observable LLM runtime look like a LangChain chat model.