from __future__ import annotations

import pytest
from somm_core.parse import extract_balanced, strip_markdown_fence


# ---------------------------------------------------------------------------
# extract_balanced
# ---------------------------------------------------------------------------


def test_extract_balanced_basic_json_object():
    assert extract_balanced('{"a":1}') == '{"a":1}'


def test_extract_balanced_nested_braces():
    assert extract_balanced('{{"a":1}}') == '{{"a":1}}'


def test_extract_balanced_prose_wrapper():
    assert extract_balanced('prefix {"k":"v"} suffix') == '{"k":"v"}'


def test_extract_balanced_no_open_char():
    assert extract_balanced("no braces here") is None


def test_extract_balanced_unmatched_open():
    assert extract_balanced("{no close") is None


def test_extract_balanced_square_brackets():
    assert extract_balanced("[1,2,3]", "[", "]") == "[1,2,3]"


# ---------------------------------------------------------------------------
# strip_markdown_fence
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_plain_json():
    assert strip_markdown_fence("```json\n{\"a\":1}\n```") == '{"a":1}'


def test_strip_markdown_fence_js():
    assert strip_markdown_fence("```js\n{}```") == "{}"


def test_strip_markdown_fence_no_fence():
    assert strip_markdown_fence('{"a":1}') == '{"a":1}'


def test_strip_markdown_fence_empty_string():
    assert strip_markdown_fence("") == ""
