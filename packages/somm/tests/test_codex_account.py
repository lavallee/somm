from somm.codex_account import weekly_quota


def test_weekly_window_is_selected_by_duration_not_position():
    result = {"rateLimits": {"primary": {"usedPercent": 91, "windowDurationMins": 10080, "resetsAt": 3000},
                             "secondary": {"usedPercent": 1, "windowDurationMins": 300, "resetsAt": 2000}}}
    quota = weekly_quota(result, now=1000)
    assert quota.used_percent == 91 and quota.resets_at == 3000


def test_five_hour_quota_is_not_mistaken_for_weekly_usage():
    result = {"rateLimits": {"primary": {"usedPercent": 99, "windowDurationMins": 300, "resetsAt": 2000}}}
    assert weekly_quota(result, now=1000).used_percent is None


def test_invalid_or_expired_weekly_data_is_unknown():
    for used, reset in ((101, 3000), (True, 3000), (3, 900), (3, None)):
        result = {"rateLimitsByLimitId": {"codex": {"secondary": {
            "usedPercent": used, "windowDurationMins": 10080, "resetsAt": reset}}}}
        assert weekly_quota(result, now=1000).used_percent is None


def test_supported_rpc_handshake_and_override_binary(tmp_path, monkeypatch):
    from somm.codex_account import account_rpc
    binary = tmp_path / "codex"
    binary.write_text('''#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "initialize":
        print(json.dumps({"id": request["id"], "result": {}}), flush=True)
    elif request.get("method") == "account/rateLimits/read":
        print(json.dumps({"method":"notification","params":{}}), flush=True)
        print(json.dumps({"id": request["id"], "result": {"rateLimits": {"primary": {"usedPercent": 19,"windowDurationMins": 10080,"resetsAt": 2000}}}}), flush=True)
''')
    binary.chmod(0o700)
    monkeypatch.setenv("SOMM_CODEX_BIN", str(binary))
    assert weekly_quota(account_rpc("account/rateLimits/read"), now=1000).used_percent == 19


def test_failed_probe_caches_unknown_without_inventing_zero(monkeypatch):
    from somm.codex_account import WeeklyQuotaProbe
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError("synthetic invalid login")
    monkeypatch.setattr("somm.codex_account.account_rpc", fail)
    probe = WeeklyQuotaProbe()
    assert probe.read().used_percent is None
    assert probe.read().error == "quota_probe_failed"
    assert len(calls) == 1
