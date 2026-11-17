import json

from somm import vendor_usage as V


def _event(ts, used, reached=None, minutes=10080, resets=1794985745):
    return {"timestamp": ts, "type": "event_msg", "payload": {
        "type": "token_count", "info": {},
        "rate_limits": {"primary": {"used_percent": used, "window_minutes": minutes,
                                    "resets_at": resets},
                        "secondary": None, "plan_type": "pro",
                        "rate_limit_reached_type": reached,
                        "credits": {"has_credits": True, "balance": "12.5"}}}}


def _write(path, events, junk_first=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (['{"partial'] if junk_first else []) + [json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n")


def test_codex_reads_the_newest_rate_limits_event(tmp_path):
    log = tmp_path / "sessions/2026/11/13/rollout-a.jsonl"
    _write(log, [_event("2026-11-13T09:00:00Z", 2.0), {"type": "other"},
                 _event("2026-11-13T09:49:18Z", 4.0)], junk_first=True)
    [r] = V.codex_readings(tmp_path)
    assert r["provider"] == "codex-cli" and r["plan"] == "pro"
    assert r["window"] == "7d" and r["used_pct"] == 4.0
    assert r["observed_at"] == "2026-11-13T09:49:18Z"
    assert r["resets_at"].startswith("2026-11-")
    assert r["credits_balance"] == "12.5"


def test_codex_skips_sessions_without_limits_and_labels_windows(tmp_path):
    import os
    old = tmp_path / "sessions/a.jsonl"
    _write(old, [_event("2026-11-01T00:00:00Z", 50.0, minutes=300)])
    new = tmp_path / "sessions/b.jsonl"
    _write(new, [{"type": "session_meta"}])
    os.utime(old, (1, 1))
    [r] = V.codex_readings(tmp_path)
    assert r["window"] == "5h" and r["used_pct"] == 50.0


def test_codex_without_a_home_is_empty(tmp_path):
    assert V.codex_readings(tmp_path / "missing") == []


def test_minimax_needs_a_key_and_reports_both_windows(monkeypatch):
    assert V.minimax_readings({}) == []
    import somm.providers.minimax as m
    monkeypatch.setattr(m, "plan_remains", lambda key: {"interval_used_pct": 10, "weekly_used_pct": 32})
    rs = V.minimax_readings({"MINIMAX_API_KEY": "k"})
    assert [(r["window"], r["used_pct"]) for r in rs] == [("5h", 10.0), ("7d", 32.0)]


def test_one_broken_reader_does_not_hide_the_others(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "minimax_readings", lambda env: 1 / 0)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    _write(tmp_path / "sessions/a.jsonl", [_event("2026-11-13T09:49:18Z", 4.0)])
    assert [r["provider"] for r in V.vendor_readings({})] == ["codex-cli"]


def test_deepseek_balance_parses_each_currency(monkeypatch):
    monkeypatch.setattr(V, "_request", lambda url, key, **k: {"is_available": True, "balance_infos": [
        {"currency": "USD", "total_balance": "12.50", "granted_balance": "2.50", "topped_up_balance": "10.00"}]})
    [b] = V.deepseek_balance({"DEEPSEEK_API_KEY": "k"})
    assert (b["provider"], b["balance"], b["granted"], b["available"]) == ("deepseek", 12.5, 2.5, True)
    assert V.deepseek_balance({}) == []


def test_openrouter_balance_is_purchased_minus_used(monkeypatch):
    monkeypatch.setattr(V, "_request", lambda url, key, **k: {"data": {"total_credits": 20, "total_usage": 7.25}})
    [b] = V.openrouter_balance({"OPENROUTER_API_KEY": "k"})
    assert b["balance"] == 12.75


def test_runpod_balance_and_spend_rate(monkeypatch):
    seen = {}

    def fake(url, key, body=None, **k):
        seen["body"] = body
        return {"data": {"myself": {"clientBalance": 41.2, "currentSpendPerHr": 0.69, "spendLimit": 80}}}

    monkeypatch.setattr(V, "_request", fake)
    [b] = V.runpod_balance({"RUNPOD_API_KEY": "k"})
    assert (b["balance"], b["spend_per_hr"]) == (41.2, 0.69)
    assert "clientBalance" in seen["body"]["query"]


def test_failed_balance_read_is_reported_not_hidden(monkeypatch):
    def boom(url, key, **k):
        raise OSError("HTTP 401")

    monkeypatch.setattr(V, "_request", boom)
    out = V.balances({"DEEPSEEK_API_KEY": "k"})
    assert out == [{"provider": "deepseek", "error": "OSError: HTTP 401", "observed_at": out[0]["observed_at"]}]


def test_plans_json_without_declared_plans_still_reports_vendor_and_balances(tmp_path, monkeypatch, capsys):
    import json as _json

    from somm.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    for k in ("MINIMAX_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    _write(tmp_path / "codex/sessions/a.jsonl", [_event("2026-11-13T09:49:18Z", 4.0)])
    monkeypatch.chdir(tmp_path)
    try:
        main(["plans", "--json"])
    except SystemExit as e:
        assert e.code in (0, None)
    out = _json.loads(capsys.readouterr().out)
    assert out["limits"] == [] and out["balances"] == []
    assert out["vendor"][0]["provider"] == "codex-cli"
