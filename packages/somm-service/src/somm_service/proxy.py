"""Provider-compatible HTTP routes over ``SommLLM``.

Three wire formats, one dispatch path:

  POST /v1/messages          Anthropic Messages
  POST /v1/chat/completions  OpenAI Chat Completions
  POST /v1/responses         OpenAI Responses (common subset)

Each route translates its request into one ``SommLLM.agenerate(messages=...)``
call (``SommLLM.astream(..., events=True)`` when the request streams) and
translates the result back. Routing, pins,
cooldowns, workload policy and ``somm/auto``, the data boundary, the budget
gate, cost, telemetry and the error taxonomy are the client's; nothing here
re-implements them. Fields that clients send routinely but the client cannot
pass through yet (sampling controls, stop sequences, parallel-call hints) are
accepted, not applied, and named in an ``x-somm-ignored`` response header. A
field whose absence would change what the caller gets back in kind (``n`` > 1,
logprobs, server tools, stored state) is refused with a 400 naming it.

The model string selects the route:
  "somm/auto"          the workload's adaptive policy picks provider and model
  "<provider>/<model>" pinned to that somm provider ("openrouter/openai/gpt-4o")
  "<model>"            pinned to the route's default provider: anthropic for
                       /v1/messages, openai for the OpenAI routes

Headers: X-Somm-Workload names a pre-registered workload (absent: the
catch-all ``proxy_default``); X-Somm-Project writes under another project.

Streaming forwards the client's events as they arrive, rendered as the
wire's own event sequence (text, tool calls, signed thinking on the Anthropic
route, reasoning on Chat Completions, usage at the end). Nothing is sent
until the route has produced its first answer event, so a failure before it
is a normal error response; a failure after it is the wire's in-stream error
event. Input tokens are known only when the stream ends: the Anthropic route
reports them on ``message_delta`` (``message_start`` says 0).
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

from somm.client import _failure_error
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommBudgetExceeded,
    SommDeferred,
    SommError,
    SommInsufficientCredit,
    SommPrivacyViolation,
    SommProviderError,
    SommProviderNotConfigured,
    SommRateLimited,
    SommRefused,
    SommTimeout,
    describe_error,
)
from somm.events import ReasoningDelta, StreamDone, StreamStart, TextDelta, ToolCallDelta
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from somm_service.http_limits import PayloadTooLarge, read_bounded_json

# Calls from harness CLIs that don't know about somm still get recorded,
# bucketed into one catch-all workload instead of going untracked.
_DEFAULT_WORKLOAD = "proxy_default"
# Output cap when an OpenAI-wire request names none (Anthropic requires one).
# The client's own default (256) would truncate ordinary chat answers.
_DEFAULT_MAX_TOKENS = 4096


class ProxyWorkloadError(ValueError):
    pass


class WireError(ValueError):
    """The request uses a wire feature this proxy does not support (400)."""


# ---------------------------------------------------------------------------
# Shared helpers


def split_model(model: str, default_provider: str) -> tuple[str | None, str]:
    """Model string → (provider pin, model). See the module docstring."""
    if model == "somm/auto":
        return None, model
    if "/" in model:
        provider, _, rest = model.partition("/")
        return provider, rest
    return default_provider, model


def _check_fields(body: dict, supported: set[str], ignorable: set[str] = frozenset(),
                  ignored: list[str] | None = None, prefix: str = "") -> None:
    """Refuse unknown fields; note the ignorable ones present in ``ignored``."""
    extra = sorted(set(body) - supported - ignorable)
    if extra:
        raise WireError(
            f"unsupported field(s) {', '.join(prefix + k for k in extra)}: the somm proxy "
            f"serves {', '.join(sorted(supported))}"
        )
    if ignored is not None:
        ignored.extend(prefix + k for k in sorted(set(body) & set(ignorable)))


def _thinking(value: Any, ignored: list[str]) -> None:
    """``thinking`` has no client argument. Adaptive thinking is what reasoning
    providers do by default; anything else (a ``budget_tokens``, ``disabled``)
    is not applied and is reported as ignored. Use reasoning effort instead."""
    if value is not None and value != {"type": "adaptive"}:
        ignored.append("thinking")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") in ("text", "input_text", "output_text")
        )
    return ""


def _system_text(system: Any) -> str:
    """Anthropic ``system``: a string or text blocks (cache_control is not carried)."""
    if isinstance(system, list):
        return "\n".join(_text_of([b]) for b in system if _text_of([b]))
    return system or ""


def _json_object(arguments: Any, where: str) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError) as exc:
        raise WireError(f"{where}: arguments must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise WireError(f"{where}: arguments must be a JSON object")
    return parsed


def _append_tool_result(messages: list[dict], block: dict) -> None:
    """Consecutive tool results share one user turn, as the neutral shape wants."""
    last = messages[-1] if messages else None
    if (
        last is not None and last["role"] == "user" and isinstance(last["content"], list)
        and all(b.get("type") == "tool_result" for b in last["content"])
    ):
        last["content"].append(block)
    else:
        messages.append({"role": "user", "content": [block]})


def _arguments_json(tc) -> str:
    return tc.arguments_raw or json.dumps(tc.arguments)


def _raw_message(result) -> dict:
    """The provider's own chat-completion message, when the answer came as one
    (OpenAI-compatible providers and ChatGPT). It keeps tool-call argument
    strings and reasoning continuation fields byte-for-byte."""
    raw = result.raw if isinstance(result.raw, dict) else {}
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def _served_model(requested: str, result) -> str:
    # Echo what the caller asked for (SDKs compare it); somm/auto reports the
    # model that actually served.
    return result.model if requested == "somm/auto" else requested


def _hex(call_id: str) -> str:
    return call_id.replace("-", "")


# ---------------------------------------------------------------------------
# Errors: one taxonomy, three envelopes


_STATUS: list[tuple[type, int, str]] = [
    (SommDeferred, 429, "work_deferred"),
    (SommPrivacyViolation, 403, "permission_error"),
    (SommProviderNotConfigured, 400, "invalid_request_error"),
    (SommBadRequest, 400, "invalid_request_error"),
    (SommAuthError, 401, "authentication_error"),
    (SommInsufficientCredit, 402, "billing_error"),
    (SommRateLimited, 429, "rate_limit_error"),
    (SommBudgetExceeded, 429, "rate_limit_error"),
    (SommTimeout, 504, "timeout_error"),
]


def _failure(exc: SommError) -> tuple[int, dict, dict]:
    """(status, error payload, headers) for the typed error the client raised.

    The client raises a provider failure as the adapter's own exception, so
    whatever the adapter attached (retry timing, provider codes, the upstream
    status) reaches the wire.
    """
    detail = describe_error(exc)
    detail["message"] = detail["message"][:512]
    if type(exc) is SommProviderError:
        # The client's wrapper for a provider failure it could not classify:
        # an upstream fault, as 0.20.1 reported it.
        detail.update(code="SOMM_PROXY_UPSTREAM_ERROR", retryable=True)
    status, error_type = 502, "api_error"
    for klass, code, kind in _STATUS:
        if isinstance(exc, klass):
            status, error_type = code, kind
            break
    else:
        if not detail["retryable"]:
            status, error_type = 400, "invalid_request_error"
    upstream_status = getattr(exc, "status_code", None)
    if isinstance(upstream_status, int) and 400 <= upstream_status <= 599:
        status = upstream_status
    payload: dict[str, Any] = {"type": error_type, "param": None, **detail}
    for field in ("call_id", "provider", "model"):
        if getattr(exc, field, None):
            payload[field] = getattr(exc, field)
    for field in ("provider_code", "provider_type", "provider_param", "provider_request_id"):
        value = getattr(exc, field, None)
        if isinstance(value, str) and value:
            payload[field] = value[:256]
    if payload.get("provider_param"):
        payload["param"] = payload["provider_param"]
    if isinstance(exc, SommDeferred):
        payload["routing"] = exc.decision
    headers = {"x-should-retry": "true" if detail["retryable"] else "false"}
    if exc.call_id:
        headers["x-somm-call-id"] = exc.call_id
    retry_after = detail.get("retry_after_s")
    if isinstance(retry_after, (float, int)) and math.isfinite(retry_after) and retry_after >= 0:
        headers["Retry-After"] = str(math.ceil(retry_after))
    return status, payload, headers


def _refusal(exc: SommRefused, wire: _Wire) -> Any:
    """A refusal is an answer, not a failed request: the wire reports it in
    its own shape (Anthropic ``stop_reason: refusal``, Chat Completions
    ``finish_reason: content_filter``, Responses a ``refusal`` content part
    in a completed response) with whatever text and usage the
    client attached to the error."""
    return types.SimpleNamespace(
        call_id=exc.call_id or "", text=exc.text or "", tool_calls=[], raw=None,
        stop_reason="refusal" if wire.envelope == "anthropic" else "content_filter",
        model=exc.model or "", provider=exc.provider,
        tokens_in=exc.tokens_in or 0, tokens_out=exc.tokens_out or 0,
        reasoning_content="", cache_tokens_in=None, reasoning_tokens=None,
    )


def _anthropic_error(*, error_type: str, message: str, status: int,
                     headers: dict | None = None, extra: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {**(extra or {}), "type": error_type, "message": message}},
        status_code=status, headers=headers,
    )


def _openai_error(*, error_type: str, message: str, status: int) -> JSONResponse:
    error = {"message": message, "type": error_type, "param": None, "code": None}
    return JSONResponse({"error": error}, status_code=status)


# ---------------------------------------------------------------------------
# Anthropic Messages


def _anthropic_request(body: dict) -> tuple[dict, list[str]]:
    ignored: list[str] = []
    _check_fields(
        body,
        {"model", "messages", "system", "max_tokens", "temperature", "tools", "tool_choice",
         "stream", "thinking", "output_config"},
        {"metadata", "stop_sequences", "top_p", "top_k"}, ignored,
    )
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise WireError("messages must be a list")
    _thinking(body.get("thinking"), ignored)
    call: dict[str, Any] = {
        "messages": messages,  # the client's neutral shape is Anthropic's own
        "system": _system_text(body.get("system")),
        "max_tokens": int(body.get("max_tokens") or _DEFAULT_MAX_TOKENS),
    }
    if body.get("temperature") is not None:
        call["temperature"] = body["temperature"]
    tools = []
    for tool in body.get("tools") or []:
        if tool.get("type") not in (None, "custom") or "input_schema" not in tool:
            raise WireError(f"tool {tool.get('name')!r}: only custom tools with input_schema are supported")
        tools.append({"name": tool["name"], "description": tool.get("description", ""),
                      "parameters": tool["input_schema"]})
    if tools:
        call["tools"] = tools
    choice = body.get("tool_choice")
    if choice is not None:
        kind = choice.get("type") if isinstance(choice, dict) else None
        _check_fields(choice, {"type", "name"}, {"disable_parallel_tool_use"}, ignored,
                      "tool_choice.")
        if kind in ("auto", "any", "none"):
            call["tool_choice"] = kind
        elif kind == "tool" and choice.get("name"):
            call["tool_choice"] = {"type": "tool", "name": choice["name"]}
        else:
            raise WireError("tool_choice must be auto, any, none or {type: tool, name}")
    config = body.get("output_config") or {}
    _check_fields(config, {"effort", "format"}, prefix="output_config.")
    if config.get("effort"):
        call["reasoning_effort"] = config["effort"]
    fmt = config.get("format")
    if fmt is not None:
        if not (isinstance(fmt, dict) and fmt.get("type") == "json_schema" and fmt.get("schema")):
            raise WireError("output_config.format must be {type: json_schema, schema}")
        call["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "output", "schema": fmt["schema"]}}
    return call, ignored


def _anthropic_body(result, model: str) -> dict:
    raw = result.raw if isinstance(result.raw, dict) else {}
    # Signed thinking blocks must be replayed on the next tool turn.
    content = [
        b for b in raw.get("content") or []
        if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
    ]
    if result.text:
        content.append({"type": "text", "text": result.text})
    for tc in result.tool_calls:
        content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
    stop = result.stop_reason
    if result.tool_calls:
        stop = "tool_use"
    elif stop not in ("end_turn", "max_tokens", "stop_sequence", "pause_turn", "refusal"):
        stop = "end_turn"
    return {
        "id": f"msg_{_hex(result.call_id)}", "type": "message", "role": "assistant",
        "model": _served_model(model, result), "content": content,
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": result.tokens_in, "output_tokens": result.tokens_out},
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


# ---------------------------------------------------------------------------
# OpenAI Chat Completions


def _openai_tools(tools: list) -> list[dict]:
    out = []
    for tool in tools or []:
        fn = tool.get("function") if tool.get("type") == "function" else None
        if not isinstance(fn, dict) or not fn.get("name"):
            raise WireError("only function tools are supported")
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def _openai_tool_choice(choice: Any) -> str | dict:
    if choice in ("auto", "none"):
        return choice
    if choice == "required":
        return "any"
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
    raise WireError("tool_choice must be auto, none, required or a named function")


def _chat_request(body: dict) -> tuple[dict, list[str]]:
    if body.get("n", 1) != 1:
        raise WireError("n must be 1: the somm proxy returns one choice")
    if body.get("logprobs") or "top_logprobs" in body:
        raise WireError("logprobs are not available through the somm proxy")
    ignored: list[str] = []
    _check_fields(
        body,
        {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "tools",
         "tool_choice", "response_format", "reasoning_effort", "thinking", "stream",
         "stream_options"},
        {"user", "stop", "top_p", "parallel_tool_calls", "n", "logprobs"}, ignored,
    )
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise WireError("messages must be a list")
    _thinking(body.get("thinking"), ignored)
    system: list[str] = []
    messages: list[dict] = []
    for m in raw_messages:
        role = m.get("role") if isinstance(m, dict) else None
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "user":
            # String or OpenAI parts; both provider families accept image_url.
            messages.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            text = _text_of(m.get("content"))
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if m.get("tool_calls"):
                blocks = [{"type": "text", "text": text}] if text else []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""),
                                   "input": _json_object(fn.get("arguments"), "tool_calls")})
                msg["content"] = blocks
            # Reasoning providers need these back on the next tool turn.
            for key in ("reasoning_content", "reasoning_details"):
                if m.get(key):
                    msg[key] = m[key]
            messages.append(msg)
        elif role == "tool":
            _append_tool_result(messages, {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                                           "content": m.get("content") or ""})
        else:
            raise WireError(f"unsupported message role {role!r}")
    call: dict[str, Any] = {
        "messages": messages,
        "system": "\n".join(s for s in system if s),
        "max_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens")
                          or _DEFAULT_MAX_TOKENS),
    }
    for key in ("temperature", "response_format", "reasoning_effort"):
        if key in body:
            call[key] = body[key]
    if body.get("tools"):
        call["tools"] = _openai_tools(body["tools"])
    if "tool_choice" in body:
        call["tool_choice"] = _openai_tool_choice(body["tool_choice"])
    return call, ignored


def _finish_reason(result) -> str:
    if result.tool_calls:
        return "tool_calls"
    return {"max_tokens": "length", "content_filter": "content_filter"}.get(result.stop_reason, "stop")


def _chat_body(result, model: str) -> dict:
    raw = _raw_message(result)
    message: dict[str, Any] = {"role": "assistant", "content": result.text}
    if result.tool_calls:
        message["tool_calls"] = raw.get("tool_calls") or [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": _arguments_json(tc)}}
            for tc in result.tool_calls
        ]
    if result.reasoning_content:
        message["reasoning_content"] = result.reasoning_content
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        if raw.get(key) is not None:
            message[key] = raw[key]
    body = {
        "id": f"chatcmpl-{_hex(result.call_id)}", "object": "chat.completion",
        "created": int(time.time()), "model": _served_model(model, result),
        "choices": [{"index": 0, "message": message, "finish_reason": _finish_reason(result)}],
        "usage": {"prompt_tokens": result.tokens_in, "completion_tokens": result.tokens_out,
                  "total_tokens": result.tokens_in + result.tokens_out},
    }
    recovery = result.raw.get("somm_recovery") if isinstance(result.raw, dict) else None
    if recovery:
        body["somm_recovery"] = recovery
    return body


# ---------------------------------------------------------------------------
# OpenAI Responses (common subset)


def _responses_request(body: dict) -> tuple[dict, list[str]]:
    ignored: list[str] = []
    _check_fields(
        body,
        {"model", "input", "instructions", "tools", "tool_choice", "max_output_tokens",
         "temperature", "reasoning", "text", "stream"},
        # Nothing is stored server-side, so store has nothing to change.
        {"metadata", "user", "store", "prompt_cache_key", "top_p", "parallel_tool_calls",
         "truncation"}, ignored,
    )
    raw = body.get("input")
    items = [{"role": "user", "content": raw}] if isinstance(raw, str) else raw
    if not isinstance(items, list):
        raise WireError("input must be a string or a list of items")
    system = [body["instructions"]] if body.get("instructions") else []
    messages: list[dict] = []
    for item in items:
        kind = item.get("type", "message") if isinstance(item, dict) else None
        if kind == "message":
            role, content = item.get("role"), item.get("content")
            if role in ("system", "developer"):
                system.append(_text_of(content))
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": _responses_content(content, role)})
            else:
                raise WireError(f"unsupported message role {role!r}")
        elif kind == "function_call":
            block = {"type": "tool_use", "id": item.get("call_id", ""), "name": item.get("name", ""),
                     "input": _json_object(item.get("arguments"), "function_call")}
            last = messages[-1] if messages else None
            if last is None or last["role"] != "assistant":
                last = {"role": "assistant", "content": []}
                messages.append(last)
            elif isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}] if last["content"] else []
            last["content"].append(block)
        elif kind == "function_call_output":
            output = item.get("output")
            _append_tool_result(messages, {"type": "tool_result", "tool_use_id": item.get("call_id", ""),
                                           "content": output if isinstance(output, str) else _text_of(output)})
        else:
            raise WireError(f"unsupported input item type {kind!r}")
    call: dict[str, Any] = {
        "messages": messages,
        "system": "\n".join(s for s in system if s),
        "max_tokens": int(body.get("max_output_tokens") or _DEFAULT_MAX_TOKENS),
    }
    if body.get("temperature") is not None:
        call["temperature"] = body["temperature"]
    reasoning = body.get("reasoning") or {}
    _check_fields(reasoning, {"effort"}, {"summary"}, ignored, "reasoning.")
    if reasoning.get("effort"):
        call["reasoning_effort"] = reasoning["effort"]
    tools = []
    for tool in body.get("tools") or []:
        if tool.get("type") != "function" or not tool.get("name"):
            raise WireError("only function tools are supported")
        tools.append({"name": tool["name"], "description": tool.get("description", ""),
                      "parameters": tool.get("parameters") or {"type": "object", "properties": {}}})
    if tools:
        call["tools"] = tools
    if "tool_choice" in body:
        call["tool_choice"] = _openai_tool_choice(body["tool_choice"])
    text = body.get("text") or {}
    _check_fields(text, {"format"}, {"verbosity"}, ignored, "text.")
    fmt = text.get("format")
    if fmt and fmt.get("type") == "json_schema":
        call["response_format"] = {"type": "json_schema", "json_schema": {
            k: fmt[k] for k in ("name", "schema", "strict") if k in fmt}}
    elif fmt and fmt.get("type") == "json_object":
        call["response_format"] = {"type": "json_object"}
    elif fmt and fmt.get("type") != "text":
        raise WireError("text.format must be text, json_schema or json_object")
    return call, ignored


def _responses_content(content: Any, role: str) -> Any:
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            out.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image" and role == "user" and part.get("image_url"):
            out.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
        else:
            raise WireError(f"unsupported content part {kind!r}")
    return out


def _is_refusal(result) -> bool:
    """A refusal (somm's SommRefused stop reasons) is an answer whose message
    holds a ``refusal`` content part, not ``output_text``; the response is
    still ``completed``. https://developers.openai.com/api/reference/resources/responses"""
    return result.stop_reason in ("refusal", "content_filter") and not result.tool_calls


def _responses_body(result, model: str) -> dict:
    rid = _hex(result.call_id)
    output: list[dict] = []
    if _is_refusal(result):
        output.append({"id": f"msg_{rid}", "type": "message", "status": "completed",
                       "role": "assistant",
                       "content": [{"type": "refusal", "refusal": result.text or ""}]})
    elif result.text:
        output.append({"id": f"msg_{rid}", "type": "message", "status": "completed",
                       "role": "assistant",
                       "content": [{"type": "output_text", "text": result.text, "annotations": []}]})
    for tc in result.tool_calls:
        output.append({"id": f"fc_{tc.id}", "type": "function_call", "status": "completed",
                       "call_id": tc.id, "name": tc.name, "arguments": _arguments_json(tc)})
    truncated = result.stop_reason == "max_tokens" and not result.tool_calls
    return {
        "id": f"resp_{rid}", "object": "response", "created_at": int(time.time()),
        "status": "incomplete" if truncated else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
        "error": None, "model": _served_model(model, result), "output": output,
        "usage": {
            "input_tokens": result.tokens_in,
            "input_tokens_details": {"cached_tokens": result.cache_tokens_in or 0},
            "output_tokens": result.tokens_out,
            "output_tokens_details": {"reasoning_tokens": result.reasoning_tokens or 0},
            "total_tokens": result.tokens_in + result.tokens_out,
        },
    }


# ---------------------------------------------------------------------------
# Streaming: client events → each wire's own event sequence, as they arrive


class _AnthropicStream:
    """Messages SSE. Input tokens are known only at the end, so they ride on
    ``message_delta`` (``message_start`` says 0). Reasoning is shown only as
    signed thinking blocks, which a client must be able to replay; unsigned
    reasoning (another provider's) is not shown, as before."""

    def __init__(self, start, requested: str, body: dict) -> None:
        self.model = start.model if requested == "somm/auto" else requested
        self.id = f"msg_{_hex(start.call_id)}"
        self.index = -1
        self.open: str | None = None  # "text" | "tool" | None
        self.tool = -1
        self.thinking: list[str] = []

    def begin(self) -> Iterator[str]:
        message = {"id": self.id, "type": "message", "role": "assistant", "model": self.model,
                   "content": [], "stop_reason": None, "stop_sequence": None,
                   "usage": {"input_tokens": 0, "output_tokens": 0}}
        yield _sse("message_start", {"type": "message_start", "message": message})

    def _block(self, kind: str, content_block: dict) -> Iterator[str]:
        yield from self._close()
        self.index += 1
        self.open = kind
        yield _sse("content_block_start", {"type": "content_block_start", "index": self.index,
                                           "content_block": content_block})

    def _delta(self, delta: dict) -> str:
        return _sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                            "delta": delta})

    def _close(self) -> Iterator[str]:
        if self.open is not None:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.open = None

    def feed(self, ev) -> Iterator[str]:
        if isinstance(ev, ReasoningDelta):
            self.thinking.append(ev.text)
            if ev.signature:
                yield from self._block("thinking", {"type": "thinking", "thinking": ""})
                yield self._delta({"type": "thinking_delta", "thinking": "".join(self.thinking)})
                yield self._delta({"type": "signature_delta", "signature": ev.signature})
                yield from self._close()
                self.thinking = []
        elif isinstance(ev, TextDelta):
            if self.open != "text":
                yield from self._block("text", {"type": "text", "text": ""})
            yield self._delta({"type": "text_delta", "text": ev.text})
        elif isinstance(ev, ToolCallDelta):
            if ev.index != self.tool:
                self.tool = ev.index
                yield from self._block("tool", {"type": "tool_use", "id": ev.id, "name": ev.name,
                                                "input": {}})
            if ev.arguments:
                yield self._delta({"type": "input_json_delta", "partial_json": ev.arguments})

    def end(self, result) -> Iterator[str]:
        yield from self._close()
        stop = _anthropic_body(result, self.model)["stop_reason"]
        yield _sse("message_delta", {
            "type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"input_tokens": result.tokens_in, "output_tokens": result.tokens_out}})
        yield _sse("message_stop", {"type": "message_stop"})

    def error(self, payload: dict) -> Iterator[str]:
        yield _sse("error", {"type": "error", "error": {"type": payload["type"],
                                                        "message": payload["message"]}})


class _ChatStream:
    """Chat Completions chunks, then the usage chunk when asked, then [DONE]."""

    def __init__(self, start, requested: str, body: dict) -> None:
        self.head = {"id": f"chatcmpl-{_hex(start.call_id)}", "created": int(time.time()),
                     "model": start.model if requested == "somm/auto" else requested}
        self.include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        self.seen_tools: set[int] = set()

    def _chunk(self, delta: dict, finish: str | None = None, **extra) -> str:
        data = {**self.head, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"

    def begin(self) -> Iterator[str]:
        yield self._chunk({"role": "assistant", "content": ""})

    def feed(self, ev) -> Iterator[str]:
        if isinstance(ev, TextDelta):
            yield self._chunk({"content": ev.text})
        elif isinstance(ev, ReasoningDelta) and ev.text:
            yield self._chunk({"reasoning_content": ev.text})
        elif isinstance(ev, ToolCallDelta):
            call: dict[str, Any] = {"index": ev.index, "function": {"arguments": ev.arguments}}
            if ev.index not in self.seen_tools:
                self.seen_tools.add(ev.index)
                call.update(id=ev.id, type="function")
                call["function"]["name"] = ev.name
            yield self._chunk({"tool_calls": [call]})

    def end(self, result) -> Iterator[str]:
        yield self._chunk({}, _finish_reason(result))
        if self.include_usage:
            usage = {**self.head, "object": "chat.completion.chunk", "choices": [],
                     "usage": {"prompt_tokens": result.tokens_in,
                               "completion_tokens": result.tokens_out,
                               "total_tokens": result.tokens_in + result.tokens_out}}
            yield f"data: {json.dumps(usage, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    def error(self, payload: dict) -> Iterator[str]:
        yield f"data: {json.dumps({'error': payload}, separators=(',', ':'))}\n\n"


class _ResponsesStream:
    """Responses events: one output item per text run or function call, each
    opened, filled by deltas and closed in order, then the whole response."""

    def __init__(self, start, requested: str, body: dict) -> None:
        self.rid = _hex(start.call_id)
        self.model = start.model if requested == "somm/auto" else requested
        self.seq = iter(range(1 << 30))
        self.index = -1
        self.item: dict | None = None
        self.parts: list[str] = []
        self.done: list[dict] = []  # each output item as its output_item.done said

    def _event(self, kind: str, **data) -> str:
        return _sse(kind, {"type": kind, "sequence_number": next(self.seq), **data})

    def begin(self) -> Iterator[str]:
        yield self._event("response.created", response={
            "id": f"resp_{self.rid}", "object": "response", "created_at": int(time.time()),
            "status": "in_progress", "error": None, "incomplete_details": None,
            "model": self.model, "output": []})

    def _open(self, item: dict) -> Iterator[str]:
        yield from self._close()
        self.index += 1
        self.item, self.parts = item, []
        yield self._event("response.output_item.added", output_index=self.index, item=item)
        if item["type"] == "message":
            yield self._event("response.content_part.added", output_index=self.index,
                              item_id=item["id"], content_index=0,
                              part={"type": "output_text", "text": "", "annotations": []})

    def _refusal(self, text: str) -> Iterator[str]:
        """A message whose one content part is a refusal, opened, filled and
        closed: content_part.added, refusal.delta, refusal.done,
        content_part.done (the Responses streaming-events reference)."""
        self.index += 1
        item = {"id": f"msg_{self.rid}", "type": "message", "status": "in_progress",
                "role": "assistant", "content": []}
        where = {"output_index": self.index, "item_id": item["id"], "content_index": 0}
        part = {"type": "refusal", "refusal": text}
        yield self._event("response.output_item.added", output_index=self.index, item=item)
        yield self._event("response.content_part.added", **where,
                          part={"type": "refusal", "refusal": ""})
        if text:
            yield self._event("response.refusal.delta", **where, delta=text)
        yield self._event("response.refusal.done", **where, refusal=text)
        yield self._event("response.content_part.done", **where, part=part)
        done = {**item, "status": "completed", "content": [part]}
        self.done.append(done)
        yield self._event("response.output_item.done", output_index=self.index, item=done)

    def _close(self) -> Iterator[str]:
        item = self.item
        if item is None:
            return
        whole = "".join(self.parts)
        if item["type"] == "message":
            part = {"type": "output_text", "text": whole, "annotations": []}
            yield self._event("response.output_text.done", output_index=self.index,
                              item_id=item["id"], content_index=0, text=whole, logprobs=[])
            yield self._event("response.content_part.done", output_index=self.index,
                              item_id=item["id"], content_index=0, part=part)
            done = {**item, "status": "completed", "content": [part]}
        else:
            yield self._event("response.function_call_arguments.done", output_index=self.index,
                              item_id=item["id"], name=item["name"], arguments=whole)
            done = {**item, "status": "completed", "arguments": whole}
        self.done.append(done)
        yield self._event("response.output_item.done", output_index=self.index, item=done)
        self.item = None

    def feed(self, ev) -> Iterator[str]:
        if isinstance(ev, TextDelta):
            if self.item is None or self.item["type"] != "message":
                yield from self._open({"id": f"msg_{self.rid}", "type": "message",
                                       "status": "in_progress", "role": "assistant",
                                       "content": []})
            self.parts.append(ev.text)
            yield self._event("response.output_text.delta", output_index=self.index,
                              item_id=self.item["id"], content_index=0, delta=ev.text,
                              logprobs=[])
        elif isinstance(ev, ToolCallDelta):
            if ev.id or ev.name or self.item is None or self.item["type"] != "function_call":
                yield from self._open({"id": f"fc_{ev.id}", "type": "function_call",
                                       "status": "in_progress", "call_id": ev.id,
                                       "name": ev.name, "arguments": ""})
            if ev.arguments:
                self.parts.append(ev.arguments)
                yield self._event("response.function_call_arguments.delta",
                                  output_index=self.index, item_id=self.item["id"],
                                  delta=ev.arguments)

    def end(self, result) -> Iterator[str]:
        yield from self._close()
        # A refusal is known only at the end. With nothing streamed yet it is
        # sent as a refusal part; text already sent as output_text stays so.
        if _is_refusal(result) and not any(i["type"] == "message" for i in self.done):
            yield from self._refusal(result.text or "")
        body = _responses_body(result, self.model)
        body["output"] = self.done  # the final response says what the events said
        yield self._event("response.completed" if body["status"] == "completed"
                          else "response.incomplete", response=body)

    def error(self, payload: dict) -> Iterator[str]:
        yield self._event("error", code=payload.get("code"), message=payload["message"],
                          param=payload.get("param"))


async def _first_events(events) -> list:
    """Read the stream up to its first answer event (or its end), before any
    byte is sent: a failure until then is an ordinary error response."""
    first: list = []
    async for ev in events:
        first.append(ev)
        if not isinstance(ev, StreamStart):
            break
    done = first[-1] if first else None
    if isinstance(done, StreamDone):
        failure = _failure_error(done.result, None, allow_empty=False)
        # A refusal is an answer: it streams with the wire's refusal stop.
        if failure is not None and not isinstance(failure, SommRefused):
            raise failure
    if not first or not isinstance(first[0], StreamStart):
        raise SommProviderError("the stream ended before it started")
    return first


async def _stream_wire(wire: _Wire, events, first: list, requested: str, body: dict):
    out = wire.stream(first[0], requested, body)
    try:
        for chunk in out.begin():
            yield chunk
        pending = first[1:]
        while True:
            for ev in pending:
                if isinstance(ev, StreamDone):
                    for chunk in out.end(ev.result):
                        yield chunk
                    return
                for chunk in out.feed(ev):
                    yield chunk
            pending = [await anext(events)]
    except SommError as exc:
        for chunk in out.error(_failure(exc)[1]):
            yield chunk
    except Exception as exc:  # noqa: BLE001 — an unclassified provider failure
        for chunk in out.error(_failure(SommProviderError(f"{type(exc).__name__}: {exc}"))[1]):
            yield chunk
    finally:
        await events.aclose()


# ---------------------------------------------------------------------------
# The one dispatch path


@dataclasses.dataclass(frozen=True)
class _Wire:
    default_provider: str
    parse: Callable[[dict], tuple[dict, list[str]]]
    render: Callable[[Any, str], dict]
    stream: type
    error: Callable[..., JSONResponse]
    envelope: str  # "anthropic" | "openai"


_ANTHROPIC = _Wire("anthropic", _anthropic_request, _anthropic_body, _AnthropicStream,
                   _anthropic_error, "anthropic")
_CHAT = _Wire("openai", _chat_request, _chat_body, _ChatStream, _openai_error, "openai")
_RESPONSES = _Wire("openai", _responses_request, _responses_body, _ResponsesStream,
                   _openai_error, "openai")


def _resolve_workload(repo, request: Request, project: str, cfg) -> Any:
    """Read X-Somm-Workload (or fall back to ``proxy_default``). An explicit
    name must already be registered; the catch-all is registered on first use
    with the default daily cap, and refused when budgets fail closed with no
    default cap."""
    explicit_workload = request.headers.get("x-somm-workload")
    name = explicit_workload or _DEFAULT_WORKLOAD
    wl = repo.workload_by_name(name, project)
    if wl is None:
        if explicit_workload:
            raise ProxyWorkloadError(
                f"unknown X-Somm-Workload {name!r}; pre-register proxy workloads before use"
            )
        default_cap = cfg.budget_default_cap_usd_daily
        if cfg.budget_fail_closed and default_cap is None:
            raise ProxyWorkloadError(
                "proxy_default must be pre-registered or "
                "SOMM_BUDGET_DEFAULT_CAP_USD_DAILY must be set when fail-closed budgets are enabled"
            )
        wl = repo.register_workload(name=name, project=project, budget_cap_usd_daily=default_cap)
    return wl


def _client(app, project: str):
    """One SommLLM per project, built on first use and closed at shutdown."""
    clients = app.state.llm_clients
    with _CLIENTS_LOCK:
        if project not in clients:
            from somm.client import SommLLM

            cfg = dataclasses.replace(app.state.config, project=project, inprocess_workers=False)
            clients[project] = SommLLM(config=cfg, providers=app.state.providers)
        return clients[project]


_CLIENTS_LOCK = threading.Lock()


def _ignored_header(ignored: list[str]) -> dict:
    return {"x-somm-ignored": ",".join(ignored)} if ignored else {}


def _refuse(wire: _Wire, status: int, payload: dict, headers: dict,
            ignored: list[str] = ()) -> JSONResponse:
    headers = {**headers, **_ignored_header(list(ignored))}
    if wire.envelope == "openai":
        return JSONResponse({"error": payload}, status_code=status, headers=headers)
    extra = {k: v for k, v in payload.items() if k not in ("type", "message", "param")}
    return wire.error(error_type=payload["type"], message=payload["message"], status=status,
                      headers=headers, extra=extra)


async def _serve(request: Request, wire: _Wire) -> Response:
    cfg = request.app.state.config
    repo = request.app.state.repo

    def invalid(message: str, status: int = 400) -> JSONResponse:
        return wire.error(error_type="invalid_request_error", message=message, status=status)

    try:
        body = await read_bounded_json(request, max_bytes=cfg.service_proxy_max_body_bytes)
    except PayloadTooLarge as exc:
        return invalid(str(exc), 413)
    except Exception as exc:  # noqa: BLE001 — any undecodable body is the caller's error
        return invalid(f"invalid JSON body: {exc}")
    if not isinstance(body, dict):
        return invalid("request body must be a JSON object")
    body = {k: v for k, v in body.items() if v is not None}
    requested = body.get("model")
    if not isinstance(requested, str) or not requested:
        return invalid("model is required")
    try:
        call, ignored = wire.parse(body)
    except WireError as exc:
        return invalid(str(exc))
    except (AttributeError, KeyError, TypeError) as exc:
        return invalid(f"malformed request: {exc}")

    project = request.headers.get("x-somm-project") or cfg.project
    try:
        workload = _resolve_workload(repo, request, project, cfg)
    except ProxyWorkloadError as exc:
        return invalid(str(exc), 403)

    provider, model = split_model(requested, wire.default_provider)
    if body.get("stream"):
        return await _serve_stream(request, wire, workload, project, provider, model, call,
                                   ignored, requested, body)
    try:
        llm = await run_in_threadpool(_client, request.app, project)
        result = await llm.agenerate(
            "", workload=workload.name, provider=provider, model=model,
            raise_on_failure=True, **call,
        )
    except SommRefused as exc:
        result = _refusal(exc, wire)
    except SommError as exc:
        return _refuse(wire, *_failure(exc), ignored)
    except ValueError as exc:  # an invalid workload policy
        return invalid(str(exc))

    headers = {"x-somm-call-id": result.call_id, **_ignored_header(ignored)}
    return JSONResponse(wire.render(result, requested), headers=headers)


async def _serve_stream(request: Request, wire: _Wire, workload, project: str,
                        provider: str | None, model: str, call: dict, ignored: list[str],
                        requested: str, body: dict) -> Response:
    """Forward the client's stream events as the wire's own, as they arrive.
    A failure before the first answer event is a normal error response; one
    after it is the wire's in-stream error event."""
    events = None
    try:
        llm = await run_in_threadpool(_client, request.app, project)
        events = llm.astream("", workload=workload.name, provider=provider, model=model,
                             events=True, **call)
        first = await _first_events(events)
    except SommError as exc:
        if events is not None:
            await events.aclose()
        return _refuse(wire, *_failure(exc), ignored)
    except ValueError as exc:  # an invalid workload policy
        if events is not None:
            await events.aclose()
        return wire.error(error_type="invalid_request_error", message=str(exc), status=400)
    except Exception as exc:  # noqa: BLE001 — an unclassified provider failure
        if events is not None:
            await events.aclose()
        return _refuse(wire, *_failure(SommProviderError(f"{type(exc).__name__}: {exc}")),
                       ignored)
    headers = {"x-somm-call-id": first[0].call_id, **_ignored_header(ignored),
               "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(_stream_wire(wire, events, first, requested, body),
                             media_type="text/event-stream", headers=headers)


async def messages_endpoint(request: Request) -> Response:
    """POST /v1/messages — Anthropic Messages."""
    return await _serve(request, _ANTHROPIC)


async def chat_completions_endpoint(request: Request) -> Response:
    """POST /v1/chat/completions — OpenAI Chat Completions."""
    return await _serve(request, _CHAT)


async def responses_endpoint(request: Request) -> Response:
    """POST /v1/responses — OpenAI Responses, common subset (no stored state)."""
    return await _serve(request, _RESPONSES)


async def routing_decision_endpoint(request: Request) -> Response:
    """Inspect the same workload route without running inference or storing input."""
    from somm.adaptive import decide_workload

    cfg, repo = request.app.state.config, request.app.state.repo
    try:
        body = await read_bounded_json(request, max_bytes=cfg.service_proxy_max_body_bytes)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise ValueError("messages must be a list")
        project = request.headers.get("x-somm-project") or cfg.project
        workload = _resolve_workload(repo, request, project, cfg)
        decision = await run_in_threadpool(decide_workload, workload, body["messages"])
        return JSONResponse(decision.as_dict())
    except SommDeferred as exc:
        return JSONResponse(exc.decision)
    except PayloadTooLarge:
        return _openai_error(error_type="invalid_request_error", message="request too large", status=413)
    except ProxyWorkloadError as exc:
        return _openai_error(error_type="invalid_request_error", message=str(exc), status=403)
    except ValueError as exc:
        return _openai_error(error_type="invalid_request_error", message=str(exc), status=400)
