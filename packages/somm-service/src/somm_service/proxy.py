"""POST /v1/messages — Anthropic Messages-compatible LLM proxy gateway (v1).

V1 scope (DECIDED in george, owner archie): the HYBRID — somm owns the
endpoint + budget gate + telemetry; LiteLLM's Python SDK is used as a
LIBRARY (no LiteLLM proxy server) to do the provider call + cross-provider
format translation. NON-STREAMING; streaming + /v1/chat/completions are
explicit follow-ups.

Why it exists: lets harness CLIs (claude-cli) route their LLM traffic
through somm by pointing ``ANTHROPIC_BASE_URL`` at this endpoint, so every
call is budget-gated and recorded in the SAME calls.sqlite ledger as
direct ``somm.llm()`` calls. One ledger, one policy.

Request shape (Anthropic Messages API):
  POST /v1/messages
  Headers:
    X-Somm-Workload: <workload name>   (optional; defaults to config.project default)
    X-Somm-Project:  <project name>    (optional; defaults to config.project)
  Body: standard Anthropic Messages JSON (model, messages, max_tokens,
        system, tools, tool_choice, temperature, ...).

Response: standard Anthropic Messages JSON (id, type, role, model, content,
stop_reason, usage). On budget exceeded: 429 with an Anthropic-format error
body and litellm is NOT called.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm
from somm.client import enforce_workload_budget
from somm.errors import SommBudgetExceeded
from somm_core import Outcome, cost_for_call
from somm_core.models import Call
from somm_core.parse import stable_hash
from starlette.requests import Request
from starlette.responses import JSONResponse

# Default workload name when no X-Somm-Workload header is provided. Calls
# from harness CLIs that don't know about somm still get recorded — just
# bucketed into a single catch-all workload instead of going untracked.
_DEFAULT_WORKLOAD = "proxy_default"


def _provider_from_model(model: str) -> tuple[str, str]:
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
    return "anthropic", model


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


# litellm/OpenAI finish_reason → Anthropic stop_reason.
_FINISH_TO_STOP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


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

    usage_obj = getattr(resp, "usage", None) or {}
    if isinstance(usage_obj, dict):
        tokens_in = int(usage_obj.get("prompt_tokens") or 0)
        tokens_out = int(usage_obj.get("completion_tokens") or 0)
    else:
        tokens_in = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage_obj, "completion_tokens", 0) or 0)

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


def _resolve_workload(repo, request: Request, project: str) -> Any:
    """Read X-Somm-Workload header (or fall back to default) and ensure the
    workload row exists. Auto-registers so calls from harness CLIs that
    haven't pre-registered still land in calls.sqlite (observe-mode parity
    with the library path)."""
    name = request.headers.get("x-somm-workload") or _DEFAULT_WORKLOAD
    wl = repo.workload_by_name(name, project)
    if wl is None:
        wl = repo.register_workload(name=name, project=project)
    return wl


async def messages_endpoint(request: Request) -> JSONResponse:
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
        body = await request.json()
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
    workload = _resolve_workload(repo, request, project)

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

    requested_model = params["model"]
    provider, model_only = _provider_from_model(requested_model)

    call_id = str(uuid.uuid4())
    ts = datetime.now(UTC)
    t0 = time.monotonic()
    outcome = Outcome.OK
    error_kind: str | None = None
    error_detail: str | None = None
    tokens_in = tokens_out = 0
    response_body: dict[str, Any] | None = None
    try:
        resp = litellm.completion(**params)
        response_body = _litellm_to_anthropic_response(resp, requested_model=requested_model)
        tokens_in = response_body["usage"]["input_tokens"]
        tokens_out = response_body["usage"]["output_tokens"]
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
