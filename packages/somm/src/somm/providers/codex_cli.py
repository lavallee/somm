"""Codex CLI provider — routes a workload to `codex exec` (OpenAI Codex CLI, headless).

Sibling to claude_cli: a subscription-seat CLI executor, not a metered-API provider. Same intent
(segment a workload onto a seat you already pay for) and same single-shot text-gen contract.

BEST-EFFORT / lightly exercised: `codex exec` prints the final answer to stdout but does not emit
a structured usage envelope the way `claude -p --output-format json` does, so token counts are
ESTIMATED (not reported) and cost is left to somm's pricing table. Verify the invocation against
the installed codex when its seat is live before leaning on it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator

from somm.errors import SommBadRequest, SommEmptyResponse, SommTimeout, SommUpstream5xx
from somm.providers.base import (
    ProviderHealth,
    SommChunk,
    SommModel,
    SommRequest,
    SommResponse,
)


class CodexCLIProvider:
    name = "codex-cli"

    def __init__(
        self,
        binary: str = "codex",
        default_model: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.binary = binary
        self.default_model = default_model
        self.timeout = timeout

    def _prompt_text(self, request: SommRequest) -> str:
        if request.tools or request.messages is not None or request.tool_choice is not None:
            raise SommBadRequest("codex-cli is single-shot text generation; no tools/multi-turn")
        p = request.prompt
        if not isinstance(p, str):
            p = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in p)
        return f"{request.system}\n\n{p}" if request.system else p

    def generate(self, request: SommRequest) -> SommResponse:
        model = request.model or self.default_model
        cmd = [self.binary, "exec", "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        cmd += ["-"]  # read the prompt from stdin
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, input=self._prompt_text(request), capture_output=True, text=True,
                timeout=self.timeout, cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as e:
            raise SommTimeout(f"codex exec timed out after {self.timeout:.0f}s") from e
        latency_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            raise SommUpstream5xx(f"codex exec exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-300:]}")
        text = proc.stdout.strip()
        if not text and not request.allow_empty:
            raise SommEmptyResponse("codex exec produced no output")

        # No usage envelope from `codex exec` → estimate tokens for telemetry.
        from somm_core.parse import estimate_prompt_tokens
        return SommResponse(
            text=text,
            model=model or "codex-cli",
            tokens_in=estimate_prompt_tokens(self._prompt_text(request), image_token_cost=0),
            tokens_out=estimate_prompt_tokens(text, image_token_cost=0),
            latency_ms=latency_ms,
            raw={"stdout_chars": len(text)},
        )

    def stream(self, request: SommRequest) -> Iterator[SommChunk]:
        yield SommChunk(text=self.generate(request).text, done=True)

    def health(self) -> ProviderHealth:
        path = shutil.which(self.binary)
        return ProviderHealth(available=bool(path), detail=path or f"{self.binary} not on PATH")

    def models(self) -> list[SommModel]:
        return [SommModel(name=self.default_model)] if self.default_model else []

    def estimate_tokens(self, text: str | list[dict], model: str) -> int:
        from somm_core.parse import estimate_prompt_tokens

        return estimate_prompt_tokens(text, image_token_cost=1000)
