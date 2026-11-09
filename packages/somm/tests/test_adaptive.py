from somm.adaptive import AdaptivePolicy, WeeklyQuota, choose_route


def policy():
    return AdaptivePolicy.parse({
        "fast": {"provider": "chatgpt", "model": "small"},
        "strong": {"provider": "chatgpt", "model": "large"},
        "fallback": {"provider": "minimax", "model": "MiniMax-M3"},
        "fast_patterns": [r"add .+ to (?:the )?grocer(?:y list|ies)"],
    })


def route(text, used=50, history=None):
    return choose_route(policy(), (history or []) + [{"role": "user", "content": text}],
                        WeeklyQuota(used, 2000, 1000), now=1100)


def test_ordinary_action_uses_fast_and_bio_uses_strong():
    assert route("Add milk to groceries").target.model == "small"
    assert route("Write a bio for Alex").target.model == "large"
    assert route("Add employment research to groceries").target.model == "large"


def test_short_followup_and_tool_results_do_not_demote_research():
    history = [{"role": "user", "content": "Research Alex’s career"},
               {"role": "assistant", "content": "Shall I continue?"}]
    assert route("Go for it", history=history).target.model == "large"
    messages = history + [{"role": "user", "content": [{"type": "tool_result", "content": "Hello"}]}]
    assert choose_route(policy(), messages, WeeklyQuota(20, 2000, 1000), now=1100).tier == "strong"


def test_quota_boundary_falls_back_only_simple_work():
    assert route("Add milk to groceries", 89.9).target.provider == "chatgpt"
    assert route("Add milk to groceries", 90).target.provider == "minimax"
    assert route("Write a bio", 90).deferred
    assert route("Write a bio", 90).target is None


def test_unknown_stale_or_reset_quota_cannot_spend_subscription():
    for quota in (WeeklyQuota(None), WeeklyQuota(1, 2000, 500), WeeklyQuota(1, 1050, 1000)):
        decision = choose_route(policy(), [{"role": "user", "content": "Write a bio"}], quota, now=1100)
        assert decision.deferred and decision.reason == "weekly_quota_unknown"


def test_new_explicit_simple_task_can_leave_previous_strong_task():
    history = [{"role": "user", "content": "Research a career"}]
    assert route("Add milk to groceries", history=history).tier == "fast"


def test_library_routes_alias_records_actual_model_and_preserves_pins(tmp_path, monkeypatch):
    import json
    import time

    import pytest
    from somm.adaptive import _QUOTA
    from somm.client import SommLLM
    from somm.errors import SommDeferred
    from somm.providers.base import ProviderHealth, SommResponse
    from somm_core.config import Config
    from somm_core.read import search_calls

    class Provider:
        name = "chatgpt"
        def __init__(self):
            self.requests = []
        def generate(self, request):
            self.requests.append(request)
            return SommResponse("ok", "actual-model", 10, 2, 1, cost_usd=0)
        def health(self):
            return ProviderHealth(True)
        def models(self):
            return []
    cfg = Config()
    cfg.project = "adaptive-test"
    cfg.db_dir = tmp_path / ".somm"
    cfg.spool_dir = cfg.db_dir / "spool"
    cfg.budget_fail_closed = True
    prov = Provider()
    llm = SommLLM(config=cfg, providers=[prov])
    try:
        llm.repo.register_workload(name="test", project=cfg.project, budget_cap_usd_daily=0)
        from dataclasses import asdict
        llm.set_workload_policy("test", {"adaptive": {**asdict(policy()), "fast_patterns": list(policy().fast_patterns)}})
        monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(10, int(time.time()) + 1000, time.time()))
        result = llm.generate("Write a bio", workload="test", model="somm/auto")
        assert result.model == "actual-model" and prov.requests[-1].model == "large"
        llm._writer.flush()
        call = search_calls(cfg.db_path, cfg.project)[0]
        assert json.loads(call["routing_json"])["tier"] == "strong"
        monkeypatch.setattr(_QUOTA, "read", lambda: WeeklyQuota(90, int(time.time()) + 1000, time.time()))
        with pytest.raises(SommDeferred):
            llm.generate("Write a bio", workload="test", model="somm/auto")
        assert len(prov.requests) == 1
        llm.generate("Write a bio", workload="test", model="explicit", provider="chatgpt")
        assert prov.requests[-1].model == "explicit"
    finally:
        llm.close()


def test_cross_provider_fallback_preserves_tools_without_responses_ciphertext():
    from somm.adaptive import fallback_messages
    messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}],
                 "reasoning_details": [{"type": "openai.responses.reasoning", "item": {"encrypted_content": "opaque"}}]}]
    result = fallback_messages(messages)
    assert result == [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}]
    assert messages[0]["reasoning_details"]
