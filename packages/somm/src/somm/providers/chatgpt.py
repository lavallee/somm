"""ChatGPT subscription inference through Somm's authorized Responses connection."""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from itertools import chain

import httpx

from somm.chatgpt_auth import RESOURCE, ChatGPTAuth, ChatGPTAuthError
from somm.errors import (
    SommAuthError,
    SommBadRequest,
    SommEmptyResponse,
    SommRateLimited,
    SommTimeout,
    SommUpstream5xx,
    http_error,
)
from somm.providers._http import (
    SharedClient,
    close_shared_client,
    request_timeout,
    shared_client,
)
from somm.providers._openai_compat import (
    _translate_content_to_openai,
    _translate_messages_to_openai,
)
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
    ToolCall,
)


def _raw_items(message: dict) -> list[dict]:
    return [detail["item"] for detail in message.get("reasoning_details") or []
            if isinstance(detail, dict) and detail.get("type") in {"somm.native_assistant", "openai.responses.item"}
            and isinstance(detail.get("item"), dict)
            and detail["item"].get("type") in {"message", "function_call", "reasoning"}]


def _arguments(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _replay_changed(messages: list[dict]) -> bool:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        raw = _raw_items(message)
        if not raw:
            continue
        original = [(v.get("call_id"), v.get("name"), _arguments(v.get("arguments")))
                    for v in raw if v.get("type") == "function_call"]
        normalized = [(v.get("id"), (v.get("function") or {}).get("name"),
                       _arguments((v.get("function") or {}).get("arguments")))
                      for v in message.get("tool_calls") or []]
        if original != normalized:
            return True
    return False


def _validate_tool_results(items: list[dict]) -> None:
    pending = set()
    for item in items:
        kind, call_id = item.get("type"), item.get("call_id")
        if kind == "function_call":
            if not isinstance(call_id, str) or not call_id or call_id in pending:
                raise SommBadRequest("Invalid or duplicate tool call ID in conversation history")
            pending.add(call_id)
        elif kind == "function_call_output":
            if call_id not in pending:
                raise SommBadRequest("Tool result has no matching pending call in conversation history")
            pending.remove(call_id)
    if pending:
        raise SommBadRequest("Conversation history is missing completed tool results")


def prepare_response_input(messages: list[dict]) -> tuple[list[dict], dict | None]:
    # Harnesses can split a batched function into normalized child calls. Reusing
    # its original opaque response then leaves the child results orphaned. A
    # fresh same-model request keeps the visible transcript and completed effects;
    # it never executes tools or invents missing results.
    fresh = _replay_changed(messages)
    items = _response_input(messages, fresh=fresh)
    _validate_tool_results(items)
    recovery = {"kind": "normalized_tool_replay", "provider_retries": 0,
                "tool_results_preserved": sum(v.get("type") == "function_call_output" for v in items)} if fresh else None
    return items, recovery


def response_input(messages: list[dict]) -> list[dict]:
    return prepare_response_input(messages)[0]


def _response_input(messages: list[dict], *, fresh: bool = False) -> list[dict]:
    out = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            content = message.get("content") or ""
            out.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                        "output": content if isinstance(content, str) else json.dumps(content)})
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise SommBadRequest("Unsupported conversation role for ChatGPT Responses")
        if role == "system":
            role = "developer"
        if role == "assistant":
            replay = [detail["item"] for detail in message.get("reasoning_details") or []
                      if isinstance(detail, dict) and detail.get("type") in {"somm.native_assistant", "openai.responses.item"}
                      and isinstance(detail.get("item"), dict)
                      and detail["item"].get("type") in {"message", "function_call", "reasoning"}]
            if replay and not fresh:
                out.extend(replay)
                continue
            for detail in message.get("reasoning_details") or []:
                if not fresh and isinstance(detail, dict) and detail.get("type") == "openai.responses.reasoning":
                    item = detail.get("item")
                    if isinstance(item, dict) and item.get("type") == "reasoning":
                        out.append(item)
        content = message.get("content")
        if isinstance(content, str) and content:
            item = {"role": role, "content": content}
            if role == "assistant" and fresh:
                phases = [v.get("phase") for v in replay if v.get("type") == "message" and v.get("phase")]
                if phases:
                    item["phase"] = phases[-1]
            out.append(item)
        elif isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "text":
                    blocks.append({"type": "output_text" if role == "assistant" else "input_text",
                                   "text": block.get("text", "")})
                elif block.get("type") == "image_url" and role == "user":
                    image = block["image_url"]
                    blocks.append({"type": "input_image", "image_url": image["url"],
                                   "detail": image.get("detail", "auto")})
                else:
                    raise SommBadRequest("Unsupported ChatGPT content block; refusing to drop it")
            if blocks:
                out.append({"role": role, "content": blocks})
        for call in message.get("tool_calls") or []:
            function = call["function"]
            args = function["arguments"]
            out.append({"type": "function_call", "call_id": call["id"], "name": function["name"],
                        "arguments": args if isinstance(args, str) else json.dumps(args)})
    return out


def chat_completion(response: dict) -> dict:
    content, calls = [], []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content.append(block.get("text", ""))
                elif block.get("type") == "refusal":
                    content.append(block.get("refusal", ""))
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
    message = {"role": "assistant", "content": "".join(content)}
    if calls:
        message["tool_calls"] = calls
    replay = [{"type": "somm.native_assistant", "item": item}
              for item in response.get("output", [])
              if item.get("type") in {"message", "function_call", "reasoning"}]
    if replay:
        message["reasoning_details"] = replay
    usage = response.get("usage") or {}
    return {"id": response["id"], "object": "chat.completion", "created": response.get("created_at", 0),
            "model": response["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                      "completion_tokens": usage.get("output_tokens", 0),
                      "total_tokens": usage.get("total_tokens", 0),
                      "prompt_tokens_details": usage.get("input_tokens_details", {}),
                      "completion_tokens_details": usage.get("output_tokens_details", {})}}


def response_error(event: dict, *, request_id: str | None = None, status_code: int | None = None, retry_after: str | None = None) -> Exception:
    """The error for a failure event inside the Responses stream, where there
    is no HTTP status to go by. A non-2xx HTTP answer goes through
    ``somm.errors.http_error`` like every other adapter's."""
    error = (event.get("response") or {}).get("error") or event.get("error") or event
    if not isinstance(error, dict):
        error = {}
    code = error.get("code")
    kind = error.get("type")
    label = code or kind or event.get("type") or "inference_failed"
    message = f"ChatGPT inference failed: {label}"
    if kind == "invalid_request_error" or (status_code is not None and 400 <= status_code < 500 and status_code not in {401, 403, 408, 409, 429}):
        exc = SommBadRequest(message)
    elif kind in {"authentication_error", "permission_error"} or status_code in {401, 403}:
        exc = SommAuthError(message)
    elif kind == "rate_limit_error" or status_code == 429 or code in {"subscription_sharing_usage_limit_exceeded", "subscription_sharing_usage_unavailable"}:
        exc = SommRateLimited(message)
        try:
            delay = float(retry_after)
            if 0 <= delay <= 86400:
                exc.retry_after_s = exc.cooldown_s = delay
        except (ValueError, TypeError):
            pass
    elif status_code == 408:
        exc = SommTimeout(message)
    else:
        exc = SommUpstream5xx(message)
    # Keep structured diagnostics separate from prompt-free operator messages.
    exc.provider_code = code
    exc.provider_type = kind
    exc.provider_param = error.get("param")
    exc.provider_request_id = request_id
    if status_code is not None:
        exc.status_code = status_code
    return exc


class ChatGPTProvider:
    name = "chatgpt"

    def __init__(self, *, auth: ChatGPTAuth | None = None, timeout: float = 180):
        self.auth = auth or ChatGPTAuth()
        self.timeout = timeout
        self._http = SharedClient()

    def close(self) -> None:
        """Close the pooled HTTP connections; the next call opens new ones."""
        close_shared_client(self)

    def complete_chat(self, body: dict, timeout: float | None = None) -> dict:
        items, recovery = prepare_response_input(body["messages"])
        payload = {"model": body["model"], "input": items,
                   "store": False, "stream": True, "include": ["reasoning.encrypted_content"]}
        # SIWC preview rejects max_output_tokens; the subscription route has
        # no provider-side equivalent for the Chat Completions output hint.
        if body.get("reasoning_effort"):
            payload["reasoning"] = {"effort": body["reasoning_effort"]}
        tools = []
        for tool in body.get("tools") or []:
            if tool.get("type") != "function":
                raise SommBadRequest("ChatGPT adapter currently accepts function tools only")
            function = tool["function"]
            tools.append({"type": "function", "name": function["name"],
                          "description": function.get("description", ""),
                          "parameters": function.get("parameters", {"type": "object", "properties": {}}),
                          "strict": False})
        if tools:
            payload["input"].insert(0, {"type": "additional_tools", "role": "developer", "tools": tools})
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            choice = {"type": "function", "name": choice["function"]["name"]}
        if choice is not None:
            payload["tool_choice"] = choice
        output_format = body.get("response_format")
        if output_format:
            if output_format.get("type") == "json_schema":
                output_format = {"type": "json_schema", **output_format["json_schema"]}
            payload["text"] = {"format": output_format}
        try:
            token = self.auth.access_token()
        except ChatGPTAuthError as exc:
            raise SommAuthError(str(exc)) from exc
        try:
            with shared_client(self).stream(
                "POST", RESOURCE + "/responses", json=payload,
                headers={"Authorization": "Bearer " + token, "User-Agent": "somm"},
                timeout=request_timeout(timeout or self.timeout),
            ) as response:
                if not response.is_success:
                    response.read()
                    try:
                        error_body = response.json()
                    except ValueError:
                        error_body = {}
                    error = error_body.get("error", error_body) if isinstance(error_body, dict) else {}
                    label = (error.get("code") or error.get("type")) if isinstance(error, dict) else None
                    # The message names the provider's code, never its text,
                    # which can echo the conversation.
                    raise http_error(response.status_code, response.text, provider=self.name,
                                     model=body["model"], headers=response.headers,
                                     detail=f"ChatGPT inference failed: {label or 'inference_failed'}")
                data_lines = []
                for line in chain(response.iter_lines(), [""]):
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        event = json.loads("\n".join(data_lines))
                        data_lines.clear()
                        if event.get("type") == "response.completed":
                            result = event["response"]
                            if result.get("status") != "completed" or not result.get("model"):
                                raise SommUpstream5xx("ChatGPT did not complete inference")
                            completion = chat_completion(result)
                            if recovery:
                                completion["somm_recovery"] = recovery
                            return completion
                        if event.get("type") in {"response.failed", "response.incomplete", "error"}:
                            raise response_error(event, request_id=response.headers.get("x-request-id"))
        except httpx.TimeoutException as exc:
            raise SommTimeout("ChatGPT Responses timed out") from exc
        except httpx.HTTPError as exc:
            raise SommUpstream5xx("ChatGPT Responses connection failed") from exc
        raise SommUpstream5xx("ChatGPT stream ended before response.completed")

    def generate(self, request: SommRequest) -> SommResponse:
        if not request.model:
            raise SommBadRequest("Pin a model from the signed-in ChatGPT model catalog")
        messages = (_translate_messages_to_openai(request.messages) if request.messages is not None
                    else [{"role": "user", "content": _translate_content_to_openai(request.prompt)}])
        if request.system:
            messages = [{"role": "system", "content": request.system}, *messages]
        body = {"model": request.model, "messages": messages, "max_tokens": request.max_tokens,
                "reasoning_effort": request.reasoning_effort,
                "tools": [{"type": "function", "function": t} for t in request.tools],
                "response_format": request.response_format}
        if request.tool_choice:
            from somm.providers._openai_compat import _translate_tool_choice_to_openai

            body["tool_choice"] = _translate_tool_choice_to_openai(request.tool_choice)
        started = time.monotonic()
        result = self.complete_chat(body, timeout=request.timeout)
        message = result["choices"][0]["message"]
        calls = []
        for call in message.get("tool_calls", []):
            arguments = json.loads(call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise SommBadRequest("ChatGPT returned non-object tool arguments")
            calls.append(ToolCall(call["id"], call["function"]["name"], arguments,
                                  call["function"]["arguments"]))
        if not message["content"] and not calls and not request.allow_empty:
            raise SommEmptyResponse("ChatGPT returned no text or tool calls")
        usage = result["usage"]
        return SommResponse(message["content"], result["model"], usage["prompt_tokens"],
                            usage["completion_tokens"], int((time.monotonic() - started) * 1000),
                            raw=result, tool_calls=calls, stop_reason="tool_use" if calls else "end_turn",
                            cost_usd=0.0,
                            reasoning_tokens=usage.get("completion_tokens_details", {}).get("reasoning_tokens"))

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        resp = self.generate(request)
        yield SommChunk(resp.text, done=True, model=resp.model, response=resp)

    def health(self) -> ProviderHealth:
        from somm.chatgpt_auth import PERMISSION, read_json

        record = read_json(self.auth.path)
        return ProviderHealth(PERMISSION in record.get("scopes", []), "Somm-owned ChatGPT plan connection")

    def models(self) -> list[SommModel]:
        response = shared_client(self).get(RESOURCE + "/models", timeout=request_timeout(20),
                                           headers={"Authorization": "Bearer " + self.auth.access_token(), "User-Agent": "somm"})
        response.raise_for_status()
        return [SommModel(v.get("slug") or v.get("id"), capabilities=["tools", "vision"])
                for v in response.json().get("models", []) if v.get("visibility") == "list"]
