from __future__ import annotations

from somm_core.graders import (
    build_binary_judge_prompt,
    grade_response_pair,
    normalize_binary_criteria,
    parse_binary_judge_response,
    structural_score,
    text_similarity,
)


def test_text_similarity_identical_strings_are_one():
    assert text_similarity("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_text_similarity_disjoint_strings_are_low():
    assert text_similarity("quick brown fox", "slow purple bear") < 0.5


def test_text_similarity_empty_strings():
    assert text_similarity("", "") == 1.0
    assert text_similarity("hello", "") == 0.0


def test_structural_score_matching_json():
    prod = '{"name": "alice", "age": 30}'
    gold = '{"name": "alice", "age": 30}'
    assert structural_score(prod, gold) == 1.0


def test_structural_score_partial_match():
    prod = '{"name": "alice", "age": 30}'
    gold = '{"name": "bob", "age": 30}'
    score = structural_score(prod, gold)
    assert 0.0 < score < 1.0


def test_structural_score_prose_returns_none():
    assert structural_score("not json", "not json either") is None


def test_structural_score_one_parses_other_doesnt():
    assert structural_score('{"a": 1}', "prose") == 0.0


def test_grade_response_pair_runs_deterministic_graders_without_judge():
    scores = grade_response_pair('{"ok": true}', '{"ok": true}', judge={"provider": "dormant"})
    assert scores.structural_score == 1.0
    assert scores.text_similarity_score == 1.0
    assert scores.judge_score is None


def test_binary_judge_prompt_and_parse_response():
    criteria = normalize_binary_criteria(
        [
            {"name": "correctness", "description": "Facts match the gold response."},
            "completeness",
        ]
    )
    prompt = build_binary_judge_prompt(
        original_prompt="Question?",
        production_text="Candidate",
        gold_text="Gold",
        criteria=criteria,
    )
    assert "Do not use numeric ratings" in prompt
    assert "correctness" in prompt

    parsed = parse_binary_judge_response(
        '{"criteria": ['
        '{"name": "correctness", "pass": true, "reason": "matches"},'
        '{"name": "completeness", "pass": false, "reason": "missing detail"}'
        "]}",
        criteria,
    )
    assert parsed["score"] == 0.5
    assert parsed["criteria"][0]["pass"] is True
    assert parsed["criteria"][1]["reason"] == "missing detail"


def test_binary_judge_missing_criterion_counts_as_false():
    criteria = normalize_binary_criteria(["correctness", "completeness"])
    parsed = parse_binary_judge_response(
        '{"criteria": [{"name": "correctness", "pass": "yes"}]}',
        criteria,
    )
    assert parsed["score"] == 0.5
    assert parsed["criteria"][1]["pass"] is False
    assert parsed["criteria"][1]["reason"] == "missing judge result"
