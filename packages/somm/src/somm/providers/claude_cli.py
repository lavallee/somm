"""Claude CLI provider — routes a workload to `claude -p` (Claude Code, headless).

A CLI executor rather than an API provider: it runs on the user's Claude Code *subscription*
seat (no metered API key), which is the point — it lets a caller segment quality-sensitive
workloads OFF the metered/gateway hot path onto a seat they already pay for. somm still gets full
telemetry: `claude -p --output-format json` reports the model actually used, token usage, cost,
and latency, which this adapter maps onto SommResponse.

Single-shot API: no tool-calling interface, multi-turn messages, or native streaming.
Calls are lean by default: no tools, no MCP servers, no settings sources, no
persisted session, and a neutral temporary cwd (no CLAUDE.md), so a one-word
prompt costs ~3.6k input tokens instead of ~34k. Configured extra_args replace
the lean flag set (the caller then owns it). With no model given the provider
uses DEFAULT_MODEL rather than the CLI's own (most expensive) default.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator

from somm_core.parse import anthropic_prompt_tokens

from somm.errors import SommBadRequest, SommTimeout, SommUpstream5xx
from somm.providers._cli_bin import resolve_cli_binary
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)

DEFAULT_MODEL = "claude-haiku-5-5"  # same inexpensive default as the anthropic provider
_warned_default = False


def _warn_default_model_once() -> None:
    """Say, once per process, that the seat is running the fallback model.

    Before 0.23 a call with no model ran whatever the CLI was configured for,
    often the most capable model. Falling back to a small one is cheaper but
    is a quality change a caller should hear about, not discover.
    """
    global _warned_default
    if _warned_default:
        return
    _warned_default = True
    import sys

    print(
        f"somm: claude-cli call with no model; using {DEFAULT_MODEL}. "
        "Pass model= or set SOMM_CLAUDE_CLI_MODEL to choose another.",
        file=sys.stderr,
    )
_LEAN_ARGS = ["--tools", "", "--strict-mcp-config", "--setting-sources", ""]
_NO_PERSIST = "--no-session-persistence"  # absent from old CLIs; see generate()


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(
        self,
        binary: str = "claude",
        default_model: str | None = None,   # None → DEFAULT_MODEL, never the CLI's own
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary = binary
        self.default_model = default_model or DEFAULT_MODEL
        self.timeout = timeout
        # Extra CLI flags appended to every invocation. A generator-only caller can pass
        # ["--tools", "", "--strict-mcp-config", "--disable-slash-commands",
        #  "--setting-sources", "project", "--no-session-persistence"] to drop the
        # ~28k-token tool/plugin/skill preamble the CLI otherwise loads per process
        # (measured 37k -> 9k prompt tokens on a one-word prompt, 2026-09-02).
        # When set they replace the built-in lean flags.
        self.extra_args = list(extra_args or [])

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
        if not request.model and self.default_model == DEFAULT_MODEL:
            _warn_default_model_once()
        cmd = [self.binary, "-p", "--output-format", "json", "--model", model]
        if request.claude_cli_text_only:
            if self.extra_args:
                raise SommBadRequest(
                    "claude_cli_text_only cannot be combined with claude_cli_extra_args "
                    "(including SOMM_CLAUDE_CLI_ARGS); remove the extra arguments"
                )
            cmd += _LEAN_ARGS
        else:
            cmd += self.extra_args or _LEAN_ARGS
        # This provider exists to spend the SUBSCRIPTION seat. An ambient
        # ANTHROPIC_API_KEY (set for the API provider) silently takes auth
        # precedence inside the CLI and bills — or fails on — the API
        # account instead; shield the subprocess from it.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_API_KEY")
        }
        prompt = self._prompt_text(request)
        t0 = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="somm-claude-cli-") as cwd:
                def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
                    return subprocess.run(
                        argv, input=prompt, capture_output=True, text=True,
                        timeout=self.timeout, cwd=cwd, env=env,
                    )

                # A provider call is a one-shot: leave no session transcript behind.
                proc = run(cmd + [_NO_PERSIST])
                if proc.returncode != 0 and f"unknown option '{_NO_PERSIST}'" in (proc.stderr or ""):
                    proc = run(cmd)  # CLI too old to know the flag
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
        # Claude Code may run an auxiliary Haiku pass with more output than
        # a short main answer. Prefer a confirmed pin, then the model carrying
        # the main prompt (including cache reads/writes), not the longest reply.
        actual_model = model if model in model_usage else next(
            (key for key, entry in model_usage.items()
             if model and (entry or {}).get("canonicalModel") == model),
            None,
        )
        if actual_model is None:
            prompt_volumes = {
                key: sum((entry or {}).get(field, 0) or 0 for field in (
                    "inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"
                ))
                for key, entry in model_usage.items()
            }
            has_prompt_volume = any(prompt_volumes.values())
            actual_model = max(
                model_usage,
                key=lambda key: prompt_volumes[key] if has_prompt_volume
                else (model_usage[key] or {}).get("outputTokens", 0),
                default=None,
            ) or model or "claude-cli"
        # The seat reports what it actually charged; prefer it over a computed
        # price. Absent (older CLI, or an error envelope) leaves cost_usd None
        # so the client falls back to its own pricing rather than recording $0.
        reported_cost = data.get("total_cost_usd")
        try:
            reported_cost = float(reported_cost) if reported_cost is not None else None
        except (TypeError, ValueError):
            reported_cost = None
        return SommResponse(
            text=data.get("result", "") or "",
            model=actual_model,
            # Anthropic splits the prompt across uncached/cache-read/cache-write;
            # input_tokens alone reports 3 for a 6,429-token prompt.
            tokens_in=anthropic_prompt_tokens(usage),
            tokens_out=int(usage.get("output_tokens", 0)),
            latency_ms=int(data.get("duration_ms", latency_ms)),
            raw=data,
            stop_reason=data.get("stop_reason", "") or "",
            cost_usd=reported_cost,
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        # No partial streaming via the JSON one-shot; deliver the whole result once.
        resp = self.generate(request)
        yield SommChunk(text=resp.text, done=True, model=resp.model)

    def health(self) -> ProviderHealth:
        # ``binary`` is normally the path the registry resolved; a bare name
        # (direct construction) is looked up the same way the registry does.
        if os.sep in self.binary:
            ok = os.path.isfile(self.binary) and os.access(self.binary, os.X_OK)
            detail = self.binary if ok else f"{self.binary} is not an executable file"
            return ProviderHealth(available=ok, detail=detail)
        lookup = resolve_cli_binary(self.binary, explicit=None, env_var="SOMM_CLAUDE_BIN")
        return ProviderHealth(available=lookup.found, detail=lookup.detail)

    def models(self) -> list[SommModel]:
        return [SommModel(name=self.default_model)] if self.default_model else []

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        return estimate_prompt_tokens(text, image_token_cost=1000)
