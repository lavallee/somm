"""Compat shim tests — GenericLLMCompat + openai_chat_completions.

Exercises the legacy-shape APIs against a fake provider; verifies
telemetry lands and the returned shape matches expectations.
"""

from __future__ import annotations

from pathlib import Path

from somm.compat import GenericLLMCompat, OpenAIChatCompletion, openai_chat_completions
from somm.compat.generic import LegacyLLMResult
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config


class FakeProvider:
    def __init__(self, name: str = "fake", text: str = "ok"):
        self.name = name
        self.text = text
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return SommResponse(
            text=self.text,
            model=request.model or f"{self.name}-m",
            tokens_in=5,
            tokens_out=2,
            latency_ms=20,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _tmp_cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "compat-test"
    cfg.db_dir = tmp_path / ".somm"
    return cfg


# ---------------------------------------------------------------------------
# GenericLLMCompat


def test_generic_compat_matches_legacy_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p = FakeProvider("ollama")
    shim = GenericLLMCompat(project="compat-test", providers=[p])
    try:
        r = shim.generate("hi", system="be brief", max_tokens=8)
    finally:
        shim.close()

    # Legacy fields
    assert isinstance(r, LegacyLLMResult)
    assert r.text == "ok"
    assert r.provider == "ollama"
    assert r.model == "ollama-m"

    # somm extras
    assert r.call_id
    assert r.tokens_in == 5
    assert r.tokens_out == 2
    assert r.latency_ms == 20
    assert r.outcome == "ok"


def test_generic_compat_chat_splits_messages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p = FakeProvider("ollama")
    shim = GenericLLMCompat(project="compat-test", providers=[p])
    try:
        r = shim.chat(
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "thanks"},
            ],
            max_tokens=8,
        )
    finally:
        shim.close()
    assert r.text == "ok"
    assert p.calls == 1


def test_generic_compat_probe_providers_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p1 = FakeProvider("ollama")
    p1.name = "ollama"
    p2 = FakeProvider("openrouter")
    p2.name = "openrouter"
    shim = GenericLLMCompat(project="compat-test", providers=[p1, p2])
    try:
        slots = shim.probe_providers(4)
    finally:
        shim.close()
    assert len(slots) == 4
    assert set(slots) <= {"ollama", "openrouter"}


def test_generic_compat_extract_structured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p = FakeProvider("ollama", text='{"count": 3}')
    shim = GenericLLMCompat(project="compat-test", providers=[p])
    try:
        data = shim.extract_structured("p", workload="es")
    finally:
        shim.close()
    assert data == {"count": 3}


def test_generic_compat_context_manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p = FakeProvider("ollama")
    with GenericLLMCompat(project="compat-test", providers=[p]) as shim:
        r = shim.generate("hi")
    assert r.text == "ok"


def test_generic_compat_propagates_provenance_via_llm_property(tmp_path, monkeypatch):
    """Users can reach somm-native features via `.llm`."""
    import somm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "compat-test")
    p = FakeProvider("ollama")
    shim = GenericLLMCompat(project="compat-test", providers=[p])
    try:
        r = shim.generate("hi", workload="prov_test")
        # legacy result still works
        assert r.text == "ok"
        # but underlying somm.llm is available for .provenance() etc.
        # We pulled the call_id from the legacy result — construct a
        # SommResult-like stub via the shim's native path.
        native = shim.llm.generate("another", workload="prov_test_2")
        prov = somm.provenance(native)
        assert prov["call_id"] == native.call_id
    finally:
        shim.close()


# ---------------------------------------------------------------------------
# openai_chat_completions


def test_openai_shim_returns_chat_completion_shape(tmp_path, monkeypatch):
    """Response looks like OpenAI's ChatCompletion."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "oai-compat")
    # Reset module-level singleton
    from somm.compat import openai_compat as mod

    mod._llm_singleton = None

    # Patch SommLLM to use our fake provider instead of real chain
    from somm.client import SommLLM

    real_init = SommLLM.__init__

    def patched_init(self, project=None, mode=None, providers=None, config=None):
        if providers is None:
            providers = [FakeProvider("ollama")]
        return real_init(self, project=project, mode=mode, providers=providers, config=config)

    monkeypatch.setattr(SommLLM, "__init__", patched_init)

    try:
        resp = openai_chat_completions(
            model="ollama/gemma4:e4b",
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            project="oai-compat",
            workload="oai_shim",
        )
    finally:
        if mod._llm_singleton:
            mod._llm_singleton.close()
        mod._llm_singleton = None

    assert isinstance(resp, OpenAIChatCompletion)
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "ok"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.total_tokens == 7
    assert resp.somm_call_id
    assert resp.somm_provider == "ollama"
    assert resp.id.startswith("somm-")


def test_openai_shim_parses_provider_prefix(tmp_path, monkeypatch):
    """'openrouter/...' picks the openrouter provider."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMM_PROJECT", "oai-compat")
    from somm.compat import openai_compat as mod

    mod._llm_singleton = None

    p_ollama = FakeProvider("ollama")
    p_openrouter = FakeProvider("openrouter", text="from-openrouter")
    p_ollama.name = "ollama"
    p_openrouter.name = "openrouter"

    from somm.client import SommLLM

    real_init = SommLLM.__init__

    def patched_init(self, project=None, mode=None, providers=None, config=None):
        if providers is None:
            providers = [p_ollama, p_openrouter]
        return real_init(self, project=project, mode=mode, providers=providers, config=config)

    monkeypatch.setattr(SommLLM, "__init__", patched_init)

    try:
        resp = openai_chat_completions(
            model="openrouter/gemma-3-27b:free",
            messages=[{"role": "user", "content": "hi"}],
            project="oai-compat",
        )
    finally:
        if mod._llm_singleton:
            mod._llm_singleton.close()
        mod._llm_singleton = None

    assert resp.choices[0].message.content == "from-openrouter"
    assert resp.somm_provider == "openrouter"
