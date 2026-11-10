from types import SimpleNamespace

from somm.compare import compare_models, parse_model_specs


def test_parse_model_specs_forms():
    assert parse_model_specs(["a/m1,b:m2", "c", " "]) == [("a", "m1"), ("b", "m2"), ("c", "")]
    assert parse_model_specs(["/orphan"]) == []
    assert parse_model_specs(None) == []


def test_compare_reports_unconfigured_provider_without_calling():
    llm = SimpleNamespace(providers=[])
    out = compare_models(
        llm, [("nope", "m")], prompt="hi", workload="compare", max_tokens=8, temperature=0.0
    )
    assert out[0]["provider"] == "nope" and "not configured" in out[0]["error"]
