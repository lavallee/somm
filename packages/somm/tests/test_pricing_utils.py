from __future__ import annotations

from somm_core.pricing import _deep_merge


def test_deep_merge_leaf_delta_wins():
    result = _deep_merge({"a": 1}, {"a": 2})
    assert result["a"] == 2


def test_deep_merge_recurses_nested():
    result = _deep_merge({"x": {"a": 1, "b": 2}}, {"x": {"b": 3}})
    assert result == {"x": {"a": 1, "b": 3}}


def test_deep_merge_disjoint_keys():
    result = _deep_merge({"a": 1}, {"b": 2})
    assert result == {"a": 1, "b": 2}
