"""LLM proxy gateways for provider-compatible HTTP clients.

V1 scope (DECIDED in george, owner archie): the HYBRID — somm owns the
endpoint + budget gate + telemetry; LiteLLM's Python SDK is used as a
LIBRARY (no LiteLLM proxy server) to do the provider call + cross-provider
format translation.

Why it exists: lets harness CLIs (claude-cli) route their LLM traffic
through somm by pointing provider base URLs at this service, so every call
is budget-gated and recorded in the SAME calls.sqlite ledger as direct
``somm.llm()`` calls. One ledger, one policy.

Supported request shapes:
  POST /v1/messages
  POST /v1/chat/completions

  Headers:
    X-Somm-Workload: <workload name>   (optional; defaults to config.project default)
    X-Somm-Project:  <project name>    (optional; defaults to config.project)

On budget exceeded: 429 with a provider-compatible error body and litellm is
NOT called.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import litellm
from somm.client import enforce_workload_budget
from somm.errors import SommBudgetExceeded
from somm_core import Outcome, cost_for_call
from somm_core.models import Call
from somm_core.parse import stable_hash
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from somm_service.http_limits import PayloadTooLarge, read_bounded_json

# Default workload name when no X-Somm-Workload header is provided. Calls
# from harness CLIs that don't know about somm still get recorded — just
# bucketed into a single catch-all workload instead of going untracked.
_DEFAULT_WORKLOAD = "proxy_default"


class ProxyWorkloadError(ValueError):
    pass


def _provider_from_model(model: str, *, default_provider: str = "anthropic") -> tuple[str, str]:
    """Split a litellm-style model string into (provider, model_only).

    litellm accepts both bare names ("claude-haiku-4-5-20251001") and
    explicit provider prefixes ("anthropic/...", "openai/...",
    "openrouter/..."). Anthropic-style clients send bare names since
    this endpoint mimics api.anthropic.com — so default to "anthropic"
    when no prefix is present.
    """
    if "/" in model:
        provider, _, model_only = model.partition("/")
        return provider, model_only
    return default_provider, model


def _anthropic_to_litellm_params(body: dict) -> dict[str, Any]:
    """Map an Anthropic Messages request body to litellm.completion kwargs.

    Anthropic carries ``system`` as a top-level field (string or list of
    content blocks); litellm/OpenAI carries it as a leading message with
    role=system. Everything else passes through largely 1:1 — litellm's
    ``completion()`` accepts Anthropic-shaped messages content blocks
    directly (it translates internally when targeting non-Anthropic
    providers).
    """
    model = body.get("model")
    if not model:
        raise ValueError("model is required")

    messages = list(body.get("messages") or [])

    system = body.get("system")
    if system:
        # Anthropic's `system` can be a string or a list of content blocks
        # (e.g. for cache_control). Flatten the list form to plain text —
        # cross-provider system blocks are not portable across providers
        # the way text is, and v1 is intentionally narrow.
        if isinstance(system, list):
            parts: list[str] = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(p for p in parts if p)
        else:
            system_text = str(system)
        if system_text:
            messages = [{"role": "system", "content": system_text}, *messages]

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if "max_tokens" in body:
        params["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        params["temperature"] = body["temperature"]
    if "top_p" in body:
        params["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        params["stop"] = body["stop_sequences"]
    if "tools" in body and body["tools"]:
        params["tools"] = body["tools"]
    if "tool_choice" in body:
        params["tool_choice"] = body["tool_choice"]
    return params


def _openai_to_litellm_params(body: dict) -> dict[str, Any]:
    """Map an OpenAI Chat Completions request body to litellm kwargs."""
    model = body.get("model")
    if not model:
        raise ValueError("model is required")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if body.get("stream"):
        raise ValueError("streaming is not supported on /v1/chat/completions yet")

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    for key in (
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "user",
    ):
        if key in body:
            params[key] = body[key]
    return params


# litellm/OpenAI finish_reason → Anthropic stop_reason.
_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _usage_tokens(usage_obj: Any) -> tuple[int, int]:
    if isinstance(usage_obj, dict):
        return (
            int(usage_obj.get("prompt_tokens") or 0),
            int(usage_obj.get("completion_tokens") or 0),
        )
    return (
        int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        int(getattr(usage_obj, "completion_tokens", 0) or 0),
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _litellm_to_anthropic_response(
    resp: Any,
    *,
    requested_model: str,
) -> dict[str, Any]:
    """Map a litellm completion response back to an Anthropic Messages JSON body.

    litellm returns an OpenAI-ish ModelResponse with ``choices[0].message``
    (content + tool_calls), ``choices[0].finish_reason``, and ``usage`` with
    prompt/completion tokens. Anthropic Messages instead returns
    ``content`` as a list of typed blocks (``text`` and/or ``tool_use``),
    plus ``stop_reason`` and ``usage`` with ``input_tokens`` /
    ``output_tokens``.

    ``requested_model`` is echoed in the response so callers see the model
    string they asked for (Anthropic SDKs sometimes assert on the literal
    name; litellm may normalize the model id).
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise ValueError("litellm response had no choices")
    choice = choices[0]
    message = getattr(choice, "message", None) or {}

    text = ""
    tool_calls = []
    if isinstance(message, dict):
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
    else:
        text = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or []

    content_blocks: list[dict[str, Any]] = []
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in tool_calls:
        # tc may be a dict or an object with .id / .function.name / .function.arguments
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            tc_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}"
            name = fn.get("name") or ""
            args_raw = fn.get("arguments") or "{}"
        else:
            fn = getattr(tc, "function", None)
            tc_id = getattr(tc, "id", None) or f"toolu_{uuid.uuid4().hex[:16]}"
            name = getattr(fn, "name", "") if fn is not None else ""
            args_raw = getattr(fn, "arguments", "") if fn is not None else "{}"
        try:
            import json as _json
            args_parsed = _json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(args_parsed, dict):
                args_parsed = {}
        except Exception:
            args_parsed = {}
        content_blocks.append(
            {"type": "tool_use", "id": tc_id, "name": name, "input": args_parsed}
        )

    finish = getattr(choice, "finish_reason", None) or (
        choice.get("finish_reason") if isinstance(choice, dict) else None
    )
    stop_reason = _FINISH_TO_STOP.get(finish or "", "end_turn")
    # tool_use overrides whatever finish_reason litellm reported — some
    # backends report "stop" alongside tool_calls.
    if tool_calls and stop_reason != "tool_use":
        stop_reason = "tool_use"

    tokens_in, tokens_out = _usage_tokens(getattr(resp, "usage", None) or {})

    msg_id = getattr(resp, "id", None) or (
        resp.get("id") if isinstance(resp, dict) else None
    ) or f"msg_{uuid.uuid4().hex[:24]}"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }


def _litellm_to_openai_response(
    resp: Any,
    *,
    requested_model: str,
) -> dict[str, Any]:
    choices = getattr(resp, "choices", None) or (
        resp.get("choices") if isinstance(resp, dict) else None
    ) or []
    if not choices:
        raise ValueError("litellm response had no choices")

    response_choices: list[dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            finish = choice.get("finish_reason")
        else:
            message = getattr(choice, "message", None) or {}
            finish = getattr(choice, "finish_reason", None)

        if isinstance(message, dict):
            msg: dict[str, Any] = {
                "role": message.get("role") or "assistant",
                "content": message.get("content") or "",
            }
            tool_calls = message.get("tool_calls")
        else:
            msg = {
                "role": getattr(message, "role", None) or "assistant",
                "content": getattr(message, "content", None) or "",
            }
            tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            msg["tool_calls"] = _jsonable(tool_calls)
        response_choices.append(
            {
                "index": idx,
                "message": msg,
                "finish_reason": finish or "stop",
            }
        )

    tokens_in, tokens_out = _usage_tokens(getattr(resp, "usage", None) or {})
    msg_id = getattr(resp, "id", None) or (
        resp.get("id") if isinstance(resp, dict) else None
    ) or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = getattr(resp, "created", None) or (
        resp.get("created") if isinstance(resp, dict) else None
    ) or int(time.time())

    return {
        "id": msg_id,
        "object": "chat.completion",
        "created": created,
        "model": requested_model,
        "choices": response_choices,
        "usage": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        },
    }


def _anthropic_error(
    *,
    error_type: str,
    message: str,
    status: int,
) -> JSONResponse:
    """Build an Anthropic-format error body with the requested HTTP status."""
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=status,
    )


def _openai_error(
    *,
    error_type: str,
    message: str,
    status: int,
) -> JSONResponse:
    """Build an OpenAI-compatible error body with the requested HTTP status."""
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
        status_code=status,
    )


def _resolve_workload(repo, request: Request, project: str, cfg) -> Any:
    """Read X-Somm-Workload header (or fall back to default) and ensure the
    workload row exists. Auto-registers so calls from harness CLIs that
    haven't pre-registered still land in calls.sqlite (observe-mode parity
    with the library path)."""
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
        wl = repo.register_workload(
            name=name,
            project=project,
            budget_cap_usd_daily=default_cap,
        )
    return wl


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _choice0(resp: Any) -> Any | None:
    choices = getattr(resp, "choices", None) or (
        resp.get("choices") if isinstance(resp, dict) else None
    ) or []
    return choices[0] if choices else None


def _choice_value(choice: Any, key: str) -> Any:
    return choice.get(key) if isinstance(choice, dict) else getattr(choice, key, None)


def _delta_text_and_finish(chunk: Any) -> tuple[str, str | None]:
    choice = _choice0(chunk)
    if choice is None:
        return "", None
    delta = _choice_value(choice, "delta") or {}
    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)

    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        text = "".join(parts)

    finish = _choice_value(choice, "finish_reason")
    return text, finish


def _chunk_usage_tokens(chunk: Any) -> tuple[int, int]:
    usage = getattr(chunk, "usage", None) or (
        chunk.get("usage") if isinstance(chunk, dict) else None
    )
    return _usage_tokens(usage or {})


def _make_messages_streaming_response(
    *,
    repo,
    cfg,
    project: str,
    workload,
    body: dict[str, Any],
    params: dict[str, Any],
    requested_model: str,
    provider: str,
    model_only: str,
) -> StreamingResponse:
    """Return an Anthropic-compatible SSE response and write one call row."""

    def events() -> Iterator[str]:
        call_id = str(uuid.uuid4())
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        ts = datetime.now(UTC)
        t0 = time.monotonic()
        outcome = Outcome.OK
        error_kind: str | None = None
        error_detail: str | None = None
        tokens_in = tokens_out = 0
        response_text = ""
        finish_reason: str | None = None

        try:
            stream = litellm.completion(**params, stream=True)
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": requested_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            for chunk in stream:
                chunk_tokens_in, chunk_tokens_out = _chunk_usage_tokens(chunk)
                if chunk_tokens_in or chunk_tokens_out:
                    tokens_in = chunk_tokens_in
                    tokens_out = chunk_tokens_out
                text, finish = _delta_text_and_finish(chunk)
                if finish:
                    finish_reason = finish
                if not text:
                    continue
                response_text += text
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            stop_reason = _FINISH_TO_STOP.get(finish_reason or "", "end_turn")
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": tokens_out},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
        except Exception as exc:
            outcome = Outcome.UPSTREAM_ERROR
            error_kind = type(exc).__name__
            error_detail = f"{type(exc).__name__}: {exc}"[:512]
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_detail},
                },
            )
        finally:
            latency_ms = int((time.monotonic() - t0) * 1000)
            cost_usd = cost_for_call(repo, provider, model_only, tokens_in, tokens_out)
            call = Call(
                id=call_id,
                ts=ts,
                project=project,
                workload_id=workload.id,
                prompt_id=None,
                provider=provider,
                model=model_only,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                outcome=outcome,
                error_kind=error_kind,
                error_detail=error_detail,
                prompt_hash=stable_hash(body.get("messages")),
                response_hash=stable_hash(response_text),
                temperature=body.get("temperature"),
                max_tokens=body.get("max_tokens"),
                top_p=body.get("top_p"),
            )
            repo.write_call(call)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def messages_endpoint(request: Request) -> Response:
    """POST /v1/messages — Anthropic Messages-compatible proxy.

    Sequence:
      1. parse the Anthropic request body
      2. resolve workload from X-Somm-Workload (auto-register if needed)
      3. apply somm's fail-closed budget gate — on exceeded, return 429
         + Anthropic-format error and DO NOT call the provider
      4. dispatch via litellm.completion (handles provider call + format
         translation)
      5. write a telemetry row to calls.sqlite (one ledger)
      6. return the Anthropic-shaped response
    """
    cfg = request.app.state.config
    repo = request.app.state.repo

    try:
        body = await read_bounded_json(request, max_bytes=cfg.service_proxy_max_body_bytes)
    except PayloadTooLarge as exc:
        return _anthropic_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=413,
        )
    except json.JSONDecodeError as exc:
        return _anthropic_error(
            error_type="invalid_request_error",
            message=f"invalid JSON body: {exc}",
            status=400,
        )
    except Exception as exc:
        return _anthropic_error(
            error_type="invalid_request_error",
            message=f"invalid JSON body: {exc}",
            status=400,
        )
    if not isinstance(body, dict):
        return _anthropic_error(
            error_type="invalid_request_error",
            message="request body must be a JSON object",
            status=400,
        )

    project = request.headers.get("x-somm-project") or cfg.project
    try:
        workload = _resolve_workload(repo, request, project, cfg)
    except ProxyWorkloadError as exc:
        return _anthropic_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=403,
        )

    # Fail-closed budget gate — reuses the SAME helper the library uses, so
    # the proxy path and the direct somm.llm() path enforce one ceiling.
    try:
        enforce_workload_budget(
            repo,
            workload,
            fail_closed=cfg.budget_fail_closed,
            default_cap_usd_daily=cfg.budget_default_cap_usd_daily,
        )
    except SommBudgetExceeded as exc:
        # 429 mirrors Anthropic's own rate-limited shape; the message
        # carries the canonical SOMM_BUDGET_EXCEEDED text. litellm is NOT
        # invoked — no spend, no telemetry row (matches the library path).
        return _anthropic_error(
            error_type="rate_limit_error",
            message=str(exc),
            status=429,
        )

    try:
        params = _anthropic_to_litellm_params(body)
    except ValueError as exc:
        return _anthropic_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=400,
        )
    params.setdefault("timeout", cfg.http_timeout)

    requested_model = params["model"]
    provider, model_only = _provider_from_model(requested_model)

    if body.get("stream"):
        return _make_messages_streaming_response(
            repo=repo,
            cfg=cfg,
            project=project,
            workload=workload,
            body=body,
            params=params,
            requested_model=requested_model,
            provider=provider,
            model_only=model_only,
        )

    call_id = str(uuid.uuid4())
    ts = datetime.now(UTC)
    t0 = time.monotonic()
    outcome = Outcome.OK
    error_kind: str | None = None
    error_detail: str | None = None
    tokens_in = tokens_out = 0
    response_body: dict[str, Any] | None = None
    try:
        resp = await asyncio.wait_for(
            run_in_threadpool(litellm.completion, **params),
            timeout=cfg.http_timeout,
        )
        response_body = _litellm_to_anthropic_response(resp, requested_model=requested_model)
        tokens_in = response_body["usage"]["input_tokens"]
        tokens_out = response_body["usage"]["output_tokens"]
    except TimeoutError as exc:
        outcome = Outcome.TIMEOUT
        error_kind = type(exc).__name__
        error_detail = f"{type(exc).__name__}: provider call exceeded {cfg.http_timeout:.0f}s"[:512]
    except Exception as exc:
        outcome = Outcome.UPSTREAM_ERROR
        error_kind = type(exc).__name__
        error_detail = f"{type(exc).__name__}: {exc}"[:512]

    latency_ms = int((time.monotonic() - t0) * 1000)

    # Telemetry — one ledger, identical row shape to a library call.
    cost_usd = cost_for_call(repo, provider, model_only, tokens_in, tokens_out)
    response_text_for_hash = ""
    if response_body is not None:
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                response_text_for_hash += block.get("text", "")
    call = Call(
        id=call_id,
        ts=ts,
        project=project,
        workload_id=workload.id,
        # Anthropic messages bodies do not correspond to registered prompt bodies; binding is a library-path feature.
        prompt_id=None,
        provider=provider,
        model=model_only,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        outcome=outcome,
        error_kind=error_kind,
        error_detail=error_detail,
        prompt_hash=stable_hash(body.get("messages")),
        response_hash=stable_hash(response_text_for_hash),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
    )
    repo.write_call(call)

    if response_body is None:
        return _anthropic_error(
            error_type="api_error",
            message=error_detail or "upstream provider call failed",
            status=502,
        )
    return JSONResponse(response_body)


async def chat_completions_endpoint(request: Request) -> JSONResponse:
    """POST /v1/chat/completions — OpenAI-compatible chat proxy."""
    cfg = request.app.state.config
    repo = request.app.state.repo

    try:
        body = await read_bounded_json(request, max_bytes=cfg.service_proxy_max_body_bytes)
    except PayloadTooLarge as exc:
        return _openai_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=413,
        )
    except json.JSONDecodeError as exc:
        return _openai_error(
            error_type="invalid_request_error",
            message=f"invalid JSON body: {exc}",
            status=400,
        )
    except Exception as exc:
        return _openai_error(
            error_type="invalid_request_error",
            message=f"invalid JSON body: {exc}",
            status=400,
        )
    if not isinstance(body, dict):
        return _openai_error(
            error_type="invalid_request_error",
            message="request body must be a JSON object",
            status=400,
        )

    project = request.headers.get("x-somm-project") or cfg.project
    try:
        workload = _resolve_workload(repo, request, project, cfg)
    except ProxyWorkloadError as exc:
        return _openai_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=403,
        )

    try:
        enforce_workload_budget(
            repo,
            workload,
            fail_closed=cfg.budget_fail_closed,
            default_cap_usd_daily=cfg.budget_default_cap_usd_daily,
        )
    except SommBudgetExceeded as exc:
        return _openai_error(
            error_type="rate_limit_error",
            message=str(exc),
            status=429,
        )

    try:
        params = _openai_to_litellm_params(body)
    except ValueError as exc:
        return _openai_error(
            error_type="invalid_request_error",
            message=str(exc),
            status=400,
        )
    params.setdefault("timeout", cfg.http_timeout)

    requested_model = params["model"]
    provider, model_only = _provider_from_model(
        requested_model,
        default_provider="openai",
    )

    call_id = str(uuid.uuid4())
    ts = datetime.now(UTC)
    t0 = time.monotonic()
    outcome = Outcome.OK
    error_kind: str | None = None
    error_detail: str | None = None
    tokens_in = tokens_out = 0
    response_body: dict[str, Any] | None = None
    try:
        resp = await asyncio.wait_for(
            run_in_threadpool(litellm.completion, **params),
            timeout=cfg.http_timeout,
        )
        response_body = _litellm_to_openai_response(resp, requested_model=requested_model)
        usage = response_body["usage"]
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
    except TimeoutError as exc:
        outcome = Outcome.TIMEOUT
        error_kind = type(exc).__name__
        error_detail = f"{type(exc).__name__}: provider call exceeded {cfg.http_timeout:.0f}s"[
            :512
        ]
    except Exception as exc:
        outcome = Outcome.UPSTREAM_ERROR
        error_kind = type(exc).__name__
        error_detail = f"{type(exc).__name__}: {exc}"[:512]

    latency_ms = int((time.monotonic() - t0) * 1000)

    cost_usd = cost_for_call(repo, provider, model_only, tokens_in, tokens_out)
    response_text_for_hash = ""
    if response_body is not None:
        for choice in response_body.get("choices", []):
            message = choice.get("message") or {}
            response_text_for_hash += str(message.get("content") or "")
    call = Call(
        id=call_id,
        ts=ts,
        project=project,
        workload_id=workload.id,
        prompt_id=None,
        provider=provider,
        model=model_only,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        outcome=outcome,
        error_kind=error_kind,
        error_detail=error_detail,
        prompt_hash=stable_hash(body.get("messages")),
        response_hash=stable_hash(response_text_for_hash),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        top_p=body.get("top_p"),
    )
    repo.write_call(call)

    if response_body is None:
        return _openai_error(
            error_type="api_error",
            message=error_detail or "upstream provider call failed",
            status=502,
        )
    return JSONResponse(response_body)
