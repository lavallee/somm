from __future__ import annotations

from somm_core.parse import (
    extract_balanced,
    extract_cache_tokens,
    extract_citations,
    extract_json,
    prompt_id,
    strip_markdown_fence,
    strip_think_block,
    workload_id,
)

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


# ---------------------------------------------------------------------------
# strip_think_block
# ---------------------------------------------------------------------------


def test_strip_think_block_removes_block():
    assert strip_think_block('<think>reasoning</think>answer') == 'answer'


def test_strip_think_block_no_block_passthrough():
    assert strip_think_block('plain text') == 'plain text'


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


def test_extract_json_clean_dict():
    assert extract_json('{"k": 1}') == {'k': 1}


def test_extract_json_embedded_in_prose():
    result = extract_json('Sure! Here is the JSON: {"status": "ok"} as requested.')
    assert result == {'status': 'ok'}


def test_extract_json_with_think_block():
    result = extract_json('<think>reasoning</think>{"done": true}')
    assert result == {'done': True}


def test_extract_json_returns_none_for_non_json():
    assert extract_json('no json here') is None


# ---------------------------------------------------------------------------
# workload_id
# ---------------------------------------------------------------------------


def test_workload_id_is_deterministic():
    assert workload_id('ad_hoc') == workload_id('ad_hoc')


def test_workload_id_differs_by_name():
    assert workload_id('ad_hoc') != workload_id('search')


def test_workload_id_schema_affects_id():
    base = workload_id('w', input_schema=None)
    with_schema = workload_id('w', input_schema={'k': 'v'})
    assert base != with_schema


def test_workload_id_is_16_hex_chars():
    wid = workload_id('ad_hoc')
    assert len(wid) == 16
    assert all(c in '0123456789abcdef' for c in wid)


# ---------------------------------------------------------------------------
# prompt_id
# ---------------------------------------------------------------------------


def test_prompt_id_is_deterministic():
    assert prompt_id('hello') == prompt_id('hello')


def test_prompt_id_differs_by_body():
    assert prompt_id('hello') != prompt_id('world')


def test_prompt_id_is_16_hex_chars():
    pid = prompt_id('test prompt')
    assert len(pid) == 16
    assert all(c in '0123456789abcdef' for c in pid)


# ---------------------------------------------------------------------------
# telemetry extraction
# ---------------------------------------------------------------------------


def test_extract_cache_tokens_anthropic_shape():
    raw = {
        "usage": {
            "cache_read_input_tokens": 12,
            "cache_creation_input_tokens": 34,
        }
    }
    assert extract_cache_tokens(raw) == (12, 34)


def test_extract_cache_tokens_openai_shape():
    raw = {"usage": {"prompt_tokens_details": {"cached_tokens": 56}}}
    assert extract_cache_tokens(raw) == (56, None)


def test_extract_cache_tokens_empty_and_garbage_never_raises():
    assert extract_cache_tokens(None) == (None, None)
    assert extract_cache_tokens({"usage": "bad"}) == (None, None)
    assert extract_cache_tokens({"usage": {"prompt_tokens_details": "bad"}}) == (
        None,
        None,
    )


def test_extract_citations_perplexity_shapes():
    citations = [{"url": "https://example.com/a"}]
    assert extract_citations({"citations": citations}) == citations
    assert extract_citations({"search_results": citations}) == citations
    assert extract_citations({"sources": citations}) == citations


def test_extract_citations_empty_and_garbage_never_raises():
    assert extract_citations(None) is None
    assert extract_citations({"citations": "bad"}) is None
    assert extract_citations({"sources": {"url": "bad"}}) is None
