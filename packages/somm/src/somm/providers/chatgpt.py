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


def response_input(messages: list[dict]) -> list[dict]:
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
                      if isinstance(detail, dict) and detail.get("type") == "openai.responses.item"
                      and isinstance(detail.get("item"), dict)
                      and detail["item"].get("type") in {"message", "function_call", "reasoning"}]
            if replay:
                out.extend(replay)
                continue
            for detail in message.get("reasoning_details") or []:
                if isinstance(detail, dict) and detail.get("type") == "openai.responses.reasoning":
                    item = detail.get("item")
                    if isinstance(item, dict) and item.get("type") == "reasoning":
                        out.append(item)
        content = message.get("content")
        if isinstance(content, str) and content:
            out.append({"role": role, "content": content})
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
    replay = [{"type": "openai.responses.item", "item": item}
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


class ChatGPTProvider:
    name = "chatgpt"

    def __init__(self, *, auth: ChatGPTAuth | None = None, timeout: float = 180):
        self.auth = auth or ChatGPTAuth()
        self.timeout = timeout

    def complete_chat(self, body: dict) -> dict:
        payload = {"model": body["model"], "input": response_input(body["messages"]),
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
            with httpx.Client(timeout=self.timeout) as client, client.stream(
                "POST", RESOURCE + "/responses", json=payload,
                headers={"Authorization": "Bearer " + token, "User-Agent": "somm"},
            ) as response:
                if response.status_code in (401, 403):
                    raise SommAuthError("ChatGPT subscription authorization is unavailable; sign in to Somm again")
                if response.status_code == 429:
                    raise SommRateLimited("ChatGPT subscription is rate limited")
                if not response.is_success:
                    raise SommUpstream5xx(f"ChatGPT Responses returned HTTP {response.status_code}")
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
                            return chat_completion(result)
                        if event.get("type") in {"response.failed", "response.incomplete", "error"}:
                            error = (event.get("response") or {}).get("error") or event.get("error") or {}
                            code = error.get("code", "inference_failed") if isinstance(error, dict) else "inference_failed"
                            raise SommUpstream5xx(f"ChatGPT inference failed: {code}")
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
        result = self.complete_chat(body)
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
        yield SommChunk(self.generate(request).text, done=True)

    def health(self) -> ProviderHealth:
        from somm.chatgpt_auth import PERMISSION, read_json

        record = read_json(self.auth.path)
        return ProviderHealth(PERMISSION in record.get("scopes", []), "Somm-owned ChatGPT plan connection")

    def models(self) -> list[SommModel]:
        response = httpx.get(RESOURCE + "/models", timeout=20,
                            headers={"Authorization": "Bearer " + self.auth.access_token(), "User-Agent": "somm"})
        response.raise_for_status()
        return [SommModel(v.get("slug") or v.get("id"), capabilities=["tools", "vision"])
                for v in response.json().get("models", []) if v.get("visibility") == "list"]
