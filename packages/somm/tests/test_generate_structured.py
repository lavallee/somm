from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import somm.client as client_mod
from somm.client import SommLLM
from somm.errors import SommStructuredError
from somm.providers.base import ProviderHealth, SommResponse
from somm_core import Outcome, SommResult
from somm_core.config import Config


class ScriptedProvider:
    name = "fake"

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else ""
        return SommResponse(
            text=text,
            model=request.model or "fake-model",
            tokens_in=7,
            tokens_out=3,
            latency_ms=5,
            raw={"fake": True},
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True, detail="fake")

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return max(1, len(str(text)) // 4)


def _tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.project = "structured"
    cfg.mode = "observe"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    return cfg


def test_generate_structured_pydantic_returns_model_and_result(tmp_path):
    pydantic = pytest.importorskip("pydantic", minversion="2")

    class WineNote(pydantic.BaseModel):
        wine: str
        vintage: int

    provider = ScriptedProvider(['{"wine":"Riesling","vintage":2021}'])
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        obj, result = llm.generate_structured(
            "extract",
            schema=WineNote,
            workload="pydantic",
            provider="fake",
        )
    finally:
        llm.close()

    assert isinstance(obj, WineNote)
    assert obj.wine == "Riesling"
    assert obj.vintage == 2021
    assert isinstance(result, SommResult)
    assert result.provider == "fake"
    assert result.outcome == Outcome.OK
    assert "Respond with ONLY valid JSON matching this schema" in provider.requests[0].system
    assert provider.requests[0].response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "somm_structured_output",
            "schema": WineNote.model_json_schema(),
            "strict": True,
        },
    }


def test_generate_structured_json_schema_retries_with_feedback(tmp_path):
    schema = {
        "type": "object",
        "properties": {
            "wine": {"type": "string"},
            "region": {"type": "string"},
        },
        "required": ["wine", "region"],
    }
    provider = ScriptedProvider([
        '{"wine":"Riesling"}',
        '{"wine":"Riesling","region":"Mosel"}',
    ])
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        obj, result = llm.generate_structured(
            "extract",
            schema=schema,
            workload="json_schema_retry",
            system="You are precise.",
            provider="fake",
        )
    finally:
        llm.close()

    assert obj == {"wine": "Riesling", "region": "Mosel"}
    assert result.outcome == Outcome.OK
    assert len(provider.requests) == 2
    assert provider.requests[0].system.startswith("You are precise.")
    assert provider.requests[0].response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "somm_structured_output",
            "schema": schema,
            "strict": True,
        },
    }
    assert "Your previous response failed:" in provider.requests[1].system
    assert '{"wine":"Riesling"}' in provider.requests[1].system
    assert "region" in provider.requests[1].system


def test_generate_structured_callable_validator_return_propagates(tmp_path):
    provider = ScriptedProvider(['{"wine":"Barolo"}'])
    seen = []

    def validator(parsed):
        seen.append(parsed)
        return ("validated", parsed["wine"])

    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        obj, result = llm.generate_structured(
            "extract",
            schema=validator,
            workload="callable",
            provider="fake",
        )
    finally:
        llm.close()

    assert seen == [{"wine": "Barolo"}]
    assert obj == ("validated", "Barolo")
    assert result.model == "fake-model"
    assert provider.requests[0].system == ""


def test_generate_structured_total_failure_raises_clear_error(tmp_path):
    schema = {"type": "object", "required": ["wine"]}
    provider = ScriptedProvider(["not json", '{"producer":"missing wine"}'])
    llm = SommLLM(config=_tmp_config(tmp_path), providers=[provider])
    try:
        with pytest.raises(SommStructuredError) as exc_info:
            llm.generate_structured(
                "extract",
                schema=schema,
                workload="failure",
                provider="fake",
                retries=1,
            )
    finally:
        llm.close()

    message = str(exc_info.value)
    assert "SOMM_STRUCTURED_OUTPUT_FAILED" in message
    assert "Problem:" in message
    assert "Cause:" in message
    assert "Fix:" in message
    assert "Docs:" in message
    assert '{"producer":"missing wine"}' in message


def test_generate_structured_pydantic_guard_names_extra(tmp_path, monkeypatch):
    class PydanticLike:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object"}

        @classmethod
        def model_validate(cls, parsed):  # pragma: no cover - should not run
            return parsed

    monkeypatch.setattr(client_mod, "_pydantic_base_model", lambda: None)
    llm = SommLLM(
        config=_tmp_config(tmp_path),
        providers=[ScriptedProvider(['{"ok":true}'])],
    )
    try:
        with pytest.raises(ValueError) as exc_info:
            llm.generate_structured("extract", schema=PydanticLike, provider="fake")
    finally:
        llm.close()

    assert "somm[pydantic]" in str(exc_info.value)


def test_generate_structured_threads_session_parent_to_call_row(tmp_path):
    provider = ScriptedProvider(['{"wine":"Chablis"}'])
    cfg = _tmp_config(tmp_path)
    llm = SommLLM(config=cfg, providers=[provider])
    try:
        _, result = llm.generate_structured(
            "extract",
            schema={"type": "object", "required": ["wine"]},
            workload="threading",
            provider="fake",
            session_id="sess-structured",
            parent_call_id="parent-structured",
        )
        llm._writer.flush(timeout=2.0)
        with sqlite3.connect(cfg.db_path) as conn:
            row = conn.execute(
                "SELECT session_id, parent_call_id FROM calls WHERE id = ?",
                (result.call_id,),
            ).fetchone()
    finally:
        llm.close()

    assert row == ("sess-structured", "parent-structured")
