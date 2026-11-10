"""Native async client API parity and live-provider coverage."""

from __future__ import annotations

import inspect
import os
import threading
from pathlib import Path

import httpx
import pytest
from somm.client import SommLLM
from somm.errors import SommTransientError
from somm.providers.base import ProviderHealth, SommChunk, SommEmbedResponse, SommResponse
from somm_core import Outcome
from somm_core.config import Config


class FakeProvider:
    def __init__(
        self,
        name: str = "fake",
        *,
        text: str = "async response",
        responses: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.text = text
        self.responses = list(responses or [])
        self.fail = fail
        self.generate_calls = 0
        self.generate_thread_ids: list[int] = []

    def generate(self, request):
        self.generate_calls += 1
        self.generate_thread_ids.append(threading.get_ident())
        if self.fail:
            raise SommTransientError(f"{self.name} unavailable")
        text = self.responses.pop(0) if self.responses else self.text
        return SommResponse(
            text=text,
            model=request.model or f"{self.name}-model",
            tokens_in=7,
            tokens_out=3,
            latency_ms=17,
            raw={"provider": self.name},
        )

    def stream(self, request):
        for piece in ("hello ", "<think>hidden</think>", "async"):
            yield SommChunk(text=piece)
        yield SommChunk(text="", done=True)

    def health(self):
        return ProviderHealth(available=not self.fail)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


class FakeOllamaProvider(FakeProvider):
    name = "ollama"
    default_embed_model = "default-embed-model"

    def __init__(self) -> None:
        super().__init__(name="ollama")
        self.embed_thread_ids: list[int] = []

    def embed(self, request):
        self.embed_thread_ids.append(threading.get_ident())
        return SommEmbedResponse(
            embedding=[0.25, 0.5, 0.75],
            model=request.model or self.default_embed_model,
            tokens_in=4,
            latency_ms=11,
            raw={"provider": self.name},
        )


class StubGovernor:
    def __init__(self, decisions: dict[str, str]) -> None:
        self.decisions = decisions

    def decision(self, provider: str) -> str:
        return self.decisions.get(provider, "ok")


def _tmp_config(tmp_path: Path, project: str = "async-client") -> Config:
    cfg = Config()
    cfg.project = project
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


@pytest.mark.parametrize(
    ("sync_name", "async_name"),
    [
        ("generate", "agenerate"),
        ("stream", "astream"),
        ("generate_structured", "agenerate_structured"),
        ("extract_structured", "aextract_structured"),
        ("embed", "aembed"),
    ],
)
def test_async_signature_matches_sync(sync_name: str, async_name: str) -> None:
    sync_parameters = list(
        inspect.signature(getattr(SommLLM, sync_name)).parameters.values()
    )
    async_parameters = list(
        inspect.signature(getattr(SommLLM, async_name)).parameters.values()
    )

    assert async_parameters == sync_parameters


@pytest.mark.asyncio
async def test_agenerate_matches_sync_telemetry_row_shape(tmp_path: Path) -> None:
    provider = FakeProvider()
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[provider])
    main_thread_id = threading.get_ident()
    try:
        sync_result = llm.generate(
            "same request",
            workload="row_parity",
            provider="fake",
            model="fixed-model",
        )
        async_result = await llm.agenerate(
            "same request",
            workload="row_parity",
            provider="fake",
            model="fixed-model",
        )
    finally:
        llm.close()

    with llm.repo._open() as conn:
        cursor = conn.execute(
            "SELECT workload_id, provider, model, cost_usd, outcome, latency_ms "
            "FROM calls WHERE id IN (?, ?) ORDER BY id",
            (sync_result.call_id, async_result.call_id),
        )
        column_names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    assert column_names == [
        "workload_id",
        "provider",
        "model",
        "cost_usd",
        "outcome",
        "latency_ms",
    ]
    assert len(rows) == 2
    assert rows[0] == rows[1]
    assert rows[0][1:] == ("fake", "fixed-model", 0.0, "ok", 17)
    assert sync_result.outcome == async_result.outcome == Outcome.OK
    assert provider.generate_thread_ids[0] == main_thread_id
    assert provider.generate_thread_ids[1] != main_thread_id


@pytest.mark.asyncio
async def test_aembed_matches_sync_telemetry_row_shape(tmp_path: Path) -> None:
    provider = FakeOllamaProvider()
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    main_thread_id = threading.get_ident()
    try:
        sync_result = llm.embed(
            "same embedding request",
            workload="embed_row_parity",
            model="pinned-embed-model",
            session_id="session-1",
            parent_call_id="parent-1",
        )
        async_result = await llm.aembed(
            "same embedding request",
            workload="embed_row_parity",
            model="pinned-embed-model",
            session_id="session-1",
            parent_call_id="parent-1",
        )
    finally:
        llm.close()

    with llm.repo._open() as conn:
        rows = conn.execute(
            "SELECT workload_id, provider, model, tokens_in, tokens_out, latency_ms, "
            "cost_usd, outcome, prompt_hash, response_hash, session_id, parent_call_id "
            "FROM calls WHERE id IN (?, ?) ORDER BY id",
            (sync_result.call_id, async_result.call_id),
        ).fetchall()

    assert len(rows) == 2
    assert rows[0] == rows[1]
    assert rows[0][1:8] == (
        "ollama",
        "pinned-embed-model",
        4,
        0,
        11,
        0.0,
        "ok",
    )
    assert sync_result.embedding == async_result.embedding == [0.25, 0.5, 0.75]
    assert provider.embed_thread_ids[0] == main_thread_id
    assert provider.embed_thread_ids[1] != main_thread_id


@pytest.mark.asyncio
async def test_aembed_rejects_silently_overriding_configured_provider(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path)
    config.provider_order = ["remote-embeddings", "ollama"]
    ollama = FakeOllamaProvider()
    llm = SommLLM(config=config, providers=[ollama])
    try:
        with pytest.raises(ValueError, match="cannot honor configured provider 'remote-embeddings'"):
            await llm.aembed("do not reroute", workload="pinned_provider")
    finally:
        llm.close()

    assert ollama.embed_thread_ids == []


@pytest.mark.asyncio
async def test_agenerate_pin_stickiness_matches_sync(tmp_path: Path) -> None:
    """Both pin modes — sticky (default) and opt-in rescue — behave
    identically through agenerate()."""
    broken = FakeProvider("broken", fail=True)
    rescue = FakeProvider("rescue", text="rescued")
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[broken, rescue])
    try:
        sync_fallback = llm.generate(
            "route",
            workload="fallback",
            provider="broken",
            model="pinned-model",
            allow_fallback=True,
        )
        async_fallback = await llm.agenerate(
            "route",
            workload="fallback",
            provider="broken",
            model="pinned-model",
            allow_fallback=True,
        )
        sync_pinned = llm.generate(
            "route",
            workload="no_fallback",
            provider="broken",
            model="pinned-model",
        )
        async_pinned = await llm.agenerate(
            "route",
            workload="no_fallback",
            provider="broken",
            model="pinned-model",
        )
    finally:
        llm.close()

    assert (sync_fallback.provider, sync_fallback.text, sync_fallback.outcome) == (
        "rescue",
        "rescued",
        Outcome.OK,
    )
    assert (
        async_fallback.provider,
        async_fallback.text,
        async_fallback.outcome,
    ) == (sync_fallback.provider, sync_fallback.text, sync_fallback.outcome)
    assert (sync_pinned.provider, sync_pinned.model, sync_pinned.outcome) == (
        "broken",
        "pinned-model",
        Outcome.UPSTREAM_ERROR,
    )
    assert (
        async_pinned.provider,
        async_pinned.model,
        async_pinned.outcome,
    ) == (sync_pinned.provider, sync_pinned.model, sync_pinned.outcome)


@pytest.mark.asyncio
async def test_agenerate_plan_governor_matches_sync(tmp_path: Path) -> None:
    deferred = FakeProvider("metered", text="deferred")
    preferred = FakeProvider("in-pace", text="preferred")
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[deferred, preferred])
    llm.router.plan_governor = StubGovernor({"metered": "defer", "in-pace": "ok"})
    try:
        sync_result = llm.generate("route", workload="plan_governor")
        async_result = await llm.agenerate("route", workload="plan_governor")
    finally:
        llm.close()

    assert sync_result.provider == async_result.provider == "in-pace"
    assert deferred.generate_calls == 0
    assert preferred.generate_calls == 2


@pytest.mark.asyncio
async def test_async_structured_variants_share_generate_pipeline(tmp_path: Path) -> None:
    provider = FakeProvider(
        responses=[
            '{"kind":"strict"}',
            '{"kind":"permissive"}',
        ]
    )
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        structured, result = await llm.agenerate_structured(
            "extract",
            schema={"type": "object", "required": ["kind"]},
            workload="strict_structured",
            provider="fake",
        )
        extracted = await llm.aextract_structured(
            "extract",
            workload="permissive_structured",
            provider="fake",
        )
    finally:
        llm.close()

    assert structured == {"kind": "strict"}
    assert result.outcome == Outcome.OK
    assert extracted == {"kind": "permissive"}
    assert provider.generate_calls == 2
    with llm.repo._open() as conn:
        rows = conn.execute("SELECT provider FROM calls ORDER BY ts").fetchall()
    assert rows == [("fake",), ("fake",)]


@pytest.mark.asyncio
async def test_astream_yields_and_writes_sync_stream_telemetry(tmp_path: Path) -> None:
    provider = FakeProvider()
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        pieces = [piece async for piece in llm.astream("stream", workload="async_stream")]
    finally:
        llm.close()

    assert "".join(pieces) == "hello async"
    with llm.repo._open() as conn:
        rows = conn.execute(
            "SELECT provider, outcome, latency_ms FROM calls ORDER BY ts"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0:2] == ("fake", "ok")
    assert rows[0][2] >= 0


def _ollama_live() -> bool:
    url = os.environ.get("SOMM_OLLAMA_URL", "http://localhost:11434")
    try:
        return httpx.get(f"{url}/api/tags", timeout=1.0).status_code == 200
    except Exception:
        return False


def _ollama_test_model() -> str | None:
    configured = os.environ.get("SOMM_OLLAMA_MODEL")
    if configured:
        return configured
    url = os.environ.get("SOMM_OLLAMA_URL", "http://localhost:11434")
    try:
        response = httpx.get(f"{url}/api/tags", timeout=1.0)
        response.raise_for_status()
        names = [
            model.get("name") or model.get("model") or ""
            for model in response.json().get("models", [])
        ]
        if "qwen2.5:7b" in names:
            return "qwen2.5:7b"
        return next((name for name in names if name and "embed" not in name.lower()), None)
    except Exception:
        return None


@pytest.mark.asyncio
@pytest.mark.skipif(not _ollama_live(), reason="no local ollama")
async def test_agenerate_against_live_ollama(tmp_path: Path) -> None:
    """Exercise the public async path end to end against a real provider."""
    from somm.providers.ollama import OllamaProvider

    model = _ollama_test_model()
    if not model:
        pytest.skip("ollama reachable but has no generation model installed")
    url = os.environ.get("SOMM_OLLAMA_URL", "http://localhost:11434")
    provider = OllamaProvider(base_url=url, default_model=model, timeout=120.0)
    llm = SommLLM(config=_tmp_config(tmp_path, project="async-live"), providers=[provider])
    try:
        result = await llm.agenerate(
            "Reply with exactly: pong",
            workload="live_async_ping",
            max_tokens=8,
            temperature=0.0,
            model=model,
            provider="ollama",
        )
    finally:
        llm.close()

    assert result.provider == "ollama"
    assert result.model == model
    assert result.outcome == Outcome.OK
    call = llm.repo.get_call(result.call_id)
    assert call is not None
    assert call.workload_id is not None
    assert (call.provider, call.model, call.outcome, call.latency_ms) == (
        result.provider,
        result.model,
        result.outcome,
        result.latency_ms,
    )
