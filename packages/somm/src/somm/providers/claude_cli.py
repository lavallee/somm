"""Claude CLI provider — routes a workload to `claude -p` (Claude Code, headless).

A CLI executor rather than an API provider: it runs on the user's Claude Code *subscription*
seat (no metered API key), which is the point — it lets a caller segment quality-sensitive
workloads OFF the metered/gateway hot path onto a seat they already pay for. somm still gets full
telemetry: `claude -p --output-format json` reports the model actually used, token usage, cost,
and latency, which this adapter maps onto SommResponse.

Single-shot text generation only: no tool-calling, no multi-turn `messages`, no native streaming
(stream() emits the whole result as one chunk). The subprocess runs in a neutral temp cwd so it
does NOT inherit a project's CLAUDE.md / tools — it's a generator, not an agent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator

from somm.errors import SommBadRequest, SommTimeout, SommUpstream5xx
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(
        self,
        binary: str = "claude",
        default_model: str | None = None,   # None → the CLI's own default model
        timeout: float = 600.0,
    ) -> None:
        self.binary = binary
        self.default_model = default_model
        self.timeout = timeout

    def _prompt_text(self, request: SommRequest) -> str:
        if request.tools or request.messages is not None or request.tool_choice is not None:
            raise SommBadRequest(
                "claude-cli is single-shot text generation; it does not support tools or "
                "multi-turn messages — route those to anthropic/openai-compat"
            )
        p = request.prompt
        if not isinstance(p, str):  # flatten content blocks to text
            p = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in p)
        return f"{request.system}\n\n{p}" if request.system else p

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        cmd = [self.binary, "-p", "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, input=self._prompt_text(request), capture_output=True, text=True,
                timeout=self.timeout, cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as e:
            raise SommTimeout(f"claude -p timed out after {self.timeout:.0f}s") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            raise SommUpstream5xx(f"claude -p exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-300:]}")
        out = proc.stdout.strip()
        try:
            data = json.loads(out)
        except json.JSONDecodeError as e:
            raise SommUpstream5xx(f"claude -p returned non-JSON: {out[:200]}") from e
        if data.get("is_error"):
            raise SommUpstream5xx(f"claude -p error: {data.get('result') or data.get('api_error_status')}")

        usage = data.get("usage") or {}
        model_usage = data.get("modelUsage") or {}
        # the model the answer mostly came from (max output tokens among models the run touched)
        actual_model = max(
            model_usage, key=lambda k: (model_usage[k] or {}).get("outputTokens", 0), default=None
        ) or model or "claude-cli"
        return SommResponse(
            text=data.get("result", "") or "",
            model=actual_model,
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
            latency_ms=int(data.get("duration_ms", latency_ms)),
            raw=data,
            stop_reason=data.get("stop_reason", "") or "",
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        # No partial streaming via the JSON one-shot; deliver the whole result once.
        yield SommChunk(text=self.generate(request).text, done=True)

    def health(self) -> ProviderHealth:
        path = shutil.which(self.binary)
        return ProviderHealth(available=bool(path), detail=path or f"{self.binary} not on PATH")

    def models(self) -> list[SommModel]:
        return [SommModel(name=self.default_model)] if self.default_model else []

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        return estimate_prompt_tokens(text, image_token_cost=1000)
