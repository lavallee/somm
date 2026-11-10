"""Provider-compatible HTTP routes over ``SommLLM``.

Three wire formats, one dispatch path:

  POST /v1/messages          Anthropic Messages
  POST /v1/chat/completions  OpenAI Chat Completions
  POST /v1/responses         OpenAI Responses (common subset)

Each route translates its request into one ``SommLLM.agenerate(messages=...)``
call and translates the result back, streamed or not. Routing, pins,
cooldowns, workload policy and ``somm/auto``, the data boundary, the budget
gate, cost, telemetry and the error taxonomy are the client's; nothing here
re-implements them. A wire field the client cannot honor is refused with a
400 naming it, never dropped silently.

The model string selects the route:
  "somm/auto"          the workload's adaptive policy picks provider and model
  "<provider>/<model>" pinned to that somm provider ("openrouter/openai/gpt-4o")
  "<model>"            pinned to the route's default provider: anthropic for
                       /v1/messages, openai for the OpenAI routes

Headers: X-Somm-Workload names a pre-registered workload (absent: the
catch-all ``proxy_default``); X-Somm-Project writes under another project.

Streaming is buffered: the client's ``stream()`` takes no ``messages`` or
``tools`` and yields only text, so a streamed request runs the same
``agenerate`` call and replays the finished answer as the wire's own event
sequence (text, tool calls and usage included).
"""

from __future__ import annotations

import dataclasses
import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import somm.errors as somm_errors
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommBudgetExceeded,
    SommDeferred,
    SommError,
    SommPrivacyViolation,
    SommProvidersExhausted,
    SommRateLimited,
    SommTimeout,
    SommTransientError,
    describe_error,
)
from somm_core.models import Outcome
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
_FAILED = {Outcome.UPSTREAM_ERROR, Outcome.TIMEOUT, Outcome.RATE_LIMIT, Outcome.EXHAUSTED}


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


def _check_fields(body: dict, supported: set[str], ignored: set[str] = frozenset()) -> None:
    extra = sorted(set(body) - supported - ignored)
    if extra:
        raise WireError(
            f"unsupported field(s) {', '.join(extra)}: the somm proxy serves "
            f"{', '.join(sorted(supported))}"
        )


def _thinking(value: Any) -> None:
    """``thinking`` has no client argument. Adaptive or enabled thinking is
    what reasoning providers do by default, so it is accepted (a
    ``budget_tokens`` is not forwarded; use reasoning effort); asking for it
    to be disabled cannot be honored and is refused."""
    if value is None:
        return
    kind = value.get("type") if isinstance(value, dict) else None
    if kind not in ("adaptive", "enabled"):
        raise WireError(
            "thinking may be adaptive or enabled; somm cannot disable thinking "
            "per request, set a reasoning effort instead"
        )


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
    (SommBadRequest, 400, "invalid_request_error"),
    (SommAuthError, 401, "authentication_error"),
    (SommRateLimited, 429, "rate_limit_error"),
    (SommBudgetExceeded, 429, "rate_limit_error"),
    (SommTimeout, 504, "timeout_error"),
]
_OUTCOME_CLASS = {
    Outcome.TIMEOUT: SommTimeout,
    Outcome.RATE_LIMIT: SommRateLimited,
    Outcome.EXHAUSTED: SommProvidersExhausted,
}


def _failure(exc: SommError | None = None, *, result=None) -> tuple[int, dict, dict]:
    """(status, error payload, headers) for a raised SommError or a failed result.

    The client raises only its own refusals (budget, boundary, deferral,
    exhaustion); a provider failure comes back as a result carrying the
    error's class name and detail, so its retry_after_s, provider codes and
    upstream status are not available here.
    """
    if exc is not None:
        cls, detail = type(exc), describe_error(exc)
        detail["message"] = detail["message"][:512]
    else:
        cls = getattr(somm_errors, result.error_kind or "", None)
        if not (isinstance(cls, type) and issubclass(cls, SommError)):
            cls = _OUTCOME_CLASS.get(result.outcome)
        message = (result.error_detail or result.error_kind or "upstream call failed")[:512]
        if cls is None:
            detail = {"code": "SOMM_PROXY_UPSTREAM_ERROR", "message": message, "retryable": True}
        else:
            retryable = issubclass(cls, (SommTransientError, SommProvidersExhausted))
            detail = {"schema_version": 1, "code": cls.code, "message": message,
                      "retryable": retryable}
    status, error_type = 502, "api_error"
    for klass, code, kind in _STATUS:
        if cls is not None and issubclass(cls, klass):
            status, error_type = code, kind
            break
    else:
        if not detail["retryable"]:
            status, error_type = 400, "invalid_request_error"
    payload: dict[str, Any] = {"type": error_type, "param": None, **detail}
    if isinstance(exc, SommDeferred):
        payload["routing"] = exc.decision
    headers = {"x-should-retry": "true" if detail["retryable"] else "false"}
    if result is not None:
        payload.update(call_id=result.call_id, provider=result.provider, model=result.model)
        headers["x-somm-call-id"] = result.call_id
    retry_after = detail.get("retry_after_s")
    if isinstance(retry_after, (float, int)) and math.isfinite(retry_after) and retry_after >= 0:
        headers["Retry-After"] = str(math.ceil(retry_after))
    return status, payload, headers


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


def _anthropic_request(body: dict) -> dict:
    _check_fields(
        body,
        {"model", "messages", "system", "max_tokens", "temperature", "tools", "tool_choice",
         "stream", "thinking", "output_config"},
        ignored={"metadata"},
    )
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise WireError("messages must be a list")
    _thinking(body.get("thinking"))
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
        if kind in ("auto", "any", "none") and set(choice) == {"type"}:
            call["tool_choice"] = kind
        elif kind == "tool" and set(choice) == {"type", "name"}:
            call["tool_choice"] = {"type": "tool", "name": choice["name"]}
        else:
            raise WireError("tool_choice must be auto, any, none or {type: tool, name}")
    config = body.get("output_config") or {}
    _check_fields(config, {"effort", "format"})
    if config.get("effort"):
        call["reasoning_effort"] = config["effort"]
    fmt = config.get("format")
    if fmt is not None:
        if not (isinstance(fmt, dict) and fmt.get("type") == "json_schema" and fmt.get("schema")):
            raise WireError("output_config.format must be {type: json_schema, schema}")
        call["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "output", "schema": fmt["schema"]}}
    return call


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


def _anthropic_events(body: dict) -> Iterator[str]:
    start = {**body, "content": [], "stop_reason": None,
             "usage": {"input_tokens": body["usage"]["input_tokens"], "output_tokens": 0}}
    yield _sse("message_start", {"type": "message_start", "message": start})
    for i, block in enumerate(body["content"]):
        kind = block["type"]
        if kind == "text":
            opened, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        elif kind == "tool_use":
            opened = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        elif kind == "thinking":
            opened = {"type": "thinking", "thinking": ""}
            deltas = [{"type": "thinking_delta", "thinking": block.get("thinking", "")},
                      {"type": "signature_delta", "signature": block.get("signature", "")}]
        else:  # redacted_thinking arrives whole
            opened, deltas = block, []
        yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                                           "content_block": opened})
        for delta in deltas:
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                                               "delta": delta})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
                                 "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                                 "usage": {"output_tokens": body["usage"]["output_tokens"]}})
    yield _sse("message_stop", {"type": "message_stop"})


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


def _chat_request(body: dict) -> dict:
    _check_fields(
        body,
        {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "tools",
         "tool_choice", "response_format", "reasoning_effort", "thinking", "stream",
         "stream_options"},
        ignored={"user"},
    )
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise WireError("messages must be a list")
    _thinking(body.get("thinking"))
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
    return call


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


def _chat_events(body: dict, include_usage: bool) -> Iterator[str]:
    head = {k: body[k] for k in ("id", "created", "model")}

    def chunk(delta: dict, finish: str | None = None) -> str:
        data = {**head, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"

    message = body["choices"][0]["message"]
    yield chunk({"role": "assistant", "content": ""})
    extras = {k: message[k] for k in ("reasoning_content", "reasoning", "reasoning_details")
              if k in message}
    if extras:
        yield chunk(extras)
    if message["content"]:
        yield chunk({"content": message["content"]})
    if message.get("tool_calls"):
        yield chunk({"tool_calls": [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]})
    yield chunk({}, body["choices"][0]["finish_reason"])
    if include_usage:
        usage = {**head, "object": "chat.completion.chunk", "choices": [], "usage": body["usage"]}
        yield f"data: {json.dumps(usage, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# OpenAI Responses (common subset)


def _responses_request(body: dict) -> dict:
    _check_fields(
        body,
        {"model", "input", "instructions", "tools", "tool_choice", "max_output_tokens",
         "temperature", "reasoning", "text", "stream"},
        # Nothing is stored server-side, so store has nothing to change.
        ignored={"metadata", "user", "store", "prompt_cache_key"},
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
    _check_fields(reasoning, {"effort"})
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
    _check_fields(text, {"format"})
    fmt = text.get("format")
    if fmt and fmt.get("type") == "json_schema":
        call["response_format"] = {"type": "json_schema", "json_schema": {
            k: fmt[k] for k in ("name", "schema", "strict") if k in fmt}}
    elif fmt and fmt.get("type") == "json_object":
        call["response_format"] = {"type": "json_object"}
    elif fmt and fmt.get("type") != "text":
        raise WireError("text.format must be text, json_schema or json_object")
    return call


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


def _responses_body(result, model: str) -> dict:
    rid = _hex(result.call_id)
    output: list[dict] = []
    if result.text:
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


def _responses_events(body: dict) -> Iterator[str]:
    seq = iter(range(1 << 30))

    def event(kind: str, **data) -> str:
        return _sse(kind, {"type": kind, "sequence_number": next(seq), **data})

    yield event("response.created", response={**body, "status": "in_progress", "output": []})
    for i, item in enumerate(body["output"]):
        if item["type"] == "message":
            part = item["content"][0]
            yield event("response.output_item.added", output_index=i,
                        item={**item, "status": "in_progress", "content": []})
            yield event("response.content_part.added", output_index=i, item_id=item["id"],
                        content_index=0, part={**part, "text": ""})
            yield event("response.output_text.delta", output_index=i, item_id=item["id"],
                        content_index=0, delta=part["text"], logprobs=[])
            yield event("response.output_text.done", output_index=i, item_id=item["id"],
                        content_index=0, text=part["text"], logprobs=[])
            yield event("response.content_part.done", output_index=i, item_id=item["id"],
                        content_index=0, part=part)
        else:
            yield event("response.output_item.added", output_index=i,
                        item={**item, "status": "in_progress", "arguments": ""})
            yield event("response.function_call_arguments.delta", output_index=i,
                        item_id=item["id"], delta=item["arguments"])
            yield event("response.function_call_arguments.done", output_index=i,
                        item_id=item["id"], name=item["name"], arguments=item["arguments"])
        yield event("response.output_item.done", output_index=i, item=item)
    yield event("response.completed" if body["status"] == "completed" else "response.incomplete",
                response=body)


# ---------------------------------------------------------------------------
# The one dispatch path


@dataclasses.dataclass(frozen=True)
class _Wire:
    default_provider: str
    parse: Callable[[dict], dict]
    render: Callable[[Any, str], dict]
    events: Callable[[dict, dict], Iterator[str]]
    error: Callable[..., JSONResponse]
    envelope: str  # "anthropic" | "openai"


_ANTHROPIC = _Wire("anthropic", _anthropic_request, _anthropic_body,
                   lambda body, _req: _anthropic_events(body), _anthropic_error, "anthropic")
_CHAT = _Wire("openai", _chat_request, _chat_body,
              lambda body, req: _chat_events(
                  body, bool((req.get("stream_options") or {}).get("include_usage"))),
              _openai_error, "openai")
_RESPONSES = _Wire("openai", _responses_request, _responses_body,
                   lambda body, _req: _responses_events(body), _openai_error, "openai")


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


def _refuse(wire: _Wire, status: int, payload: dict, headers: dict) -> JSONResponse:
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
        call = wire.parse(body)
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
    try:
        llm = await run_in_threadpool(_client, request.app, project)
        result = await llm.agenerate(
            "", workload=workload.name, provider=provider, model=model, **call
        )
    except SommError as exc:
        return _refuse(wire, *_failure(exc))
    except ValueError as exc:  # an unconfigured provider pin or an invalid policy
        return invalid(str(exc))

    if result.outcome in _FAILED:
        return _refuse(wire, *_failure(result=result))
    rendered = wire.render(result, requested)
    headers = {"x-somm-call-id": result.call_id}
    if body.get("stream"):
        return StreamingResponse(
            wire.events(rendered, body), media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(rendered, headers=headers)


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
