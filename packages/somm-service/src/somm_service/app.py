"""Starlette app serving the web admin + HTTP API.

HTTP surface:
  GET /                        HTML dashboard — status line + recs + stats
  GET /health                  JSON liveness probe
  GET /api/status              JSON dashboard summary
  GET /api/stats               JSON roll-up (per-workload × provider × model)
  GET /api/calls               JSON recent calls, filterable
  GET /api/sessions            JSON session/trace groups
  GET /api/version             JSON service + schema version
  GET /api/recommendations     JSON open recs
  POST /api/recommendations/{id}/dismiss
  POST /api/recommendations/{id}/apply
  POST /api/otlp/v1/traces     Lenient OTLP JSON trace ingest
  POST /v1/traces              Alias for OTLP JSON trace ingest
  POST /v1/messages            Anthropic Messages-compatible LLM proxy
                                (non-streaming v1; budget-gated; uses litellm
                                as a library; streaming + /v1/chat/completions
                                are explicit follow-ups)

Design tokens + a11y spec applied inline (v0.1 ships
tokens in-HTML; `packages/somm-service/web/tokens.css` lands when we extract).

`somm serve` also starts a Scheduler background thread that runs the
model_intel / shadow_eval / agent workers on their cadences.
"""

from __future__ import annotations

import contextlib
import html
import ipaddress
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from somm.recommendations import (
    apply_recommendation,
    dismiss_recommendation,
    list_recommendations,
)
from somm_core import VERSION
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.models import Call, Outcome
from somm_core.parse import stable_hash
from somm_core.repository import Repository
from somm_core.schema import current_schema_version
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from somm_service.http_limits import PayloadTooLarge, read_bounded_json
from somm_service.proxy import _anthropic_error, messages_endpoint

_CSP = "default-src 'none'; style-src 'unsafe-inline'"
_LOCAL_HEADER = "x-somm-local"
_TOKEN_ENV_VAR = "SOMM_SERVICE_TOKEN"
_READ_PROTECTED_PATHS = {
    "/",
    "/api/status",
    "/api/stats",
    "/api/calls",
    "/api/sessions",
    "/api/version",
    "/api/recommendations",
}


@dataclass(frozen=True, slots=True)
class ServiceToken:
    value: str
    path: Path
    source: str
    created: bool = False


def _service_token_path(cfg: Config) -> Path:
    return Path(cfg.db_dir) / "service_token"


def load_service_token(cfg: Config) -> ServiceToken:
    """Load the service token.

    Precedence is env > file: SOMM_SERVICE_TOKEN is intentionally first so
    tests/CI and supervised deployments can inject a secret without touching
    the local .somm token file.
    """
    token_path = _service_token_path(cfg)
    env_token = os.environ.get(_TOKEN_ENV_VAR)
    if env_token:
        return ServiceToken(value=env_token, path=token_path, source="env")

    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Claim creation with O_EXCL. The winner writes+fsyncs immediately while
    # holding the fd; a loser (FileExistsError) reads the winner's token,
    # retrying briefly to cover the microscopic window between create and
    # write. A file that stays empty is corrupt/leftover, not a live race —
    # remove it and re-claim so we never return "" (which authorizes a bare
    # `Bearer ` header).
    for _ in range(3):
        try:
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = _read_existing_token(token_path)
            if existing is not None:
                return ServiceToken(value=existing, path=token_path, source="file")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(token_path)
            continue
        token = secrets.token_urlsafe(32)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
            f.flush()
            os.fsync(f.fileno())
        return ServiceToken(value=token, path=token_path, source="file", created=True)
    # Couldn't stabilize a file token (persistent contention/corruption) —
    # fall back to an in-memory token so the service still starts securely.
    return ServiceToken(
        value=secrets.token_urlsafe(32), path=token_path, source="file", created=True
    )


def _read_existing_token(token_path: Path) -> str | None:
    """Return a non-empty token from an already-published file, or None.

    Retries briefly: a peer may have created the file (O_EXCL) an instant
    before it wrote the token, so a single read could momentarily see
    nothing. Never returns "" — an empty token would authorize a bare
    `Bearer ` header."""
    for _ in range(50):  # ~0.5s worst case
        try:
            value = token_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if value:
            return value
        time.sleep(0.01)
    return None


def _log_service_token(token: ServiceToken, *, host: str, port: int) -> None:
    if token.source == "env":
        print(f"somm service token loaded from {_TOKEN_ENV_VAR}")
        return
    if token.created:
        print(
            "somm service token created: "
            f"{token.path} "
            "(use: curl -H \"Authorization: Bearer $(cat "
            f"{token.path})\" http://{host}:{port}/v1/messages)"
        )
        return
    print(f"somm service token loaded from {token.path}")


def _host_is_loopback(host: str) -> bool:
    """True only when the request's Host header names a loopback address.

    The header-only (tokenless) auth path is meant for the local dashboard.
    Gating it on a loopback Host defeats DNS-rebinding: a rebind target
    resolves to attacker.example (or a LAN IP), whose Host is never
    loopback, so a rebound page can't ride the same-origin bypass to a
    service bound on 0.0.0.0.

    Matching is exact: `localhost`, or an IP literal whose address is
    loopback (127.0.0.0/8, ::1). A prefix check would wrongly accept
    attacker-controlled names like `127.0.0.1.attacker.com`.
    """
    if not host:
        return False
    host = host.strip()
    if host.startswith("["):  # bracketed IPv6: [addr] or [addr]:port
        end = host.find("]")
        if end == -1:
            return False  # unclosed bracket
        suffix = host[end + 1 :]
        # Suffix must be empty or a well-formed :<port> — reject
        # [::1].attacker.com, [::1]junk, etc.
        if suffix and not (suffix.startswith(":") and suffix[1:].isdigit()):
            return False
        hostname = host[1:end]
        try:  # bracket contents must be an IP literal
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False
    hostname = (host.rsplit(":", 1)[0] if host.count(":") == 1 else host).lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _origin_matches_host(origin: str, host: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and parsed.netloc == host


class LocalSecurityMiddleware:
    def __init__(self, app, *, token: str, public_read: bool = False) -> None:
        self.app = app
        self._token = token
        self._public_read = public_read

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = scope["method"]
        path = scope["path"]
        protected_read = (
            method in ("GET", "HEAD")
            and path in _READ_PROTECTED_PATHS
            and not self._public_read
        )
        protected_admin = method == "POST" and path.startswith("/api/recommendations/")
        protected_messages = method == "POST" and path == "/v1/messages"
        protected_otlp = method == "POST" and path in (
            "/api/otlp/v1/traces",
            "/v1/traces",
        )

        if protected_read or protected_admin or protected_messages or protected_otlp:
            if not self._is_authorized(headers):
                response = self._forbidden(protected_messages)
                self._set_security_headers(response)
                await response(scope, receive, send)
                return
            if (protected_messages or protected_otlp) and not self._is_json_request(headers):
                response = _anthropic_error(
                    error_type="invalid_request_error",
                    message=f"POST {path} requires Content-Type: application/json",
                    status=415,
                ) if protected_messages else JSONResponse(
                    {"ok": False, "error": f"POST {path} requires Content-Type: application/json"},
                    status_code=415,
                )
                self._set_security_headers(response)
                await response(scope, receive, send)
                return

        async def send_with_security_headers(message) -> None:
            if message["type"] == "http.response.start":
                self._set_security_headers_on_message(message)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)

    def _is_authorized(self, headers: Headers) -> bool:
        auth = headers.get("authorization", "")
        if secrets.compare_digest(auth, f"Bearer {self._token}"):
            return True

        if headers.get(_LOCAL_HEADER) != "1":
            return False

        # The header-only path is the local dashboard's; only honor it when
        # the Host is loopback, so a DNS-rebound page pointed at a service on
        # 0.0.0.0 can't ride it (its Host is attacker.example / a LAN IP).
        host = headers.get("host", "")
        if not _host_is_loopback(host):
            return False

        sec_fetch_site = headers.get("sec-fetch-site")
        if sec_fetch_site == "same-origin":
            return True

        origin = headers.get("origin")
        return bool(origin and host and _origin_matches_host(origin, host))

    @staticmethod
    def _is_json_request(headers: Headers) -> bool:
        content_type = headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        return media_type == "application/json"

    @staticmethod
    def _forbidden(anthropic_shape: bool) -> JSONResponse:
        message = (
            "authentication required: send Authorization: Bearer <token>, or for "
            "same-origin dashboard requests send X-Somm-Local: 1"
        )
        if anthropic_shape:
            return _anthropic_error(
                error_type="authentication_error",
                message=message,
                status=403,
            )
        return JSONResponse({"error": message}, status_code=403)

    @staticmethod
    def _set_security_headers(response: Response) -> None:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/html"):
            response.headers.setdefault("Content-Security-Policy", _CSP)

    @staticmethod
    def _set_security_headers_on_message(message) -> None:
        headers = MutableHeaders(scope=message)
        if "x-content-type-options" not in headers:
            headers["X-Content-Type-Options"] = "nosniff"
        if "referrer-policy" not in headers:
            headers["Referrer-Policy"] = "no-referrer"
        content_type = headers.get("content-type", "")
        if content_type.startswith("text/html") and "content-security-policy" not in headers:
            headers["Content-Security-Policy"] = _CSP


_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>somm · {project}</title>
<style>
  :root {{
    --font-sans: Inter, system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    --bg: #0a0a0a; --bg-alt: #111111;
    --fg: #e8e8e8; --fg-muted: #9ca3af;
    --border: #27272a;
    --ok: #059669; --warn: #d97706; --danger: #dc2626;
    --accent: #818cf8;
    --radius: 4px;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#fafafa; --bg-alt:#fff; --fg:#1a1a1a; --fg-muted:#6b7280; --border:#e5e7eb; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: var(--font-sans); background: var(--bg); color: var(--fg);
         margin: 0; padding: 24px 32px; line-height: 1.5; }}
  a {{ color: var(--accent); }}
  a:focus-visible, button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  header {{ display: flex; justify-content: space-between; align-items: baseline;
           border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }}
  header h1 {{ font-size: 20px; margin: 0; font-weight: 600; }}
  header .meta {{ font-family: var(--font-mono); font-size: 12px; color: var(--fg-muted); }}
    .status {{ font-size: 16px; padding: 16px; border: 1px solid var(--border);
              border-radius: var(--radius); background: var(--bg-alt); margin-bottom: 24px; }}
    .status strong {{ color: var(--ok); font-family: var(--font-mono); }}
    .status.warn strong {{ color: var(--warn); }}
    .status.err strong {{ color: var(--danger); }}
    .filters {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
                gap: 8px; margin: 0 0 20px; align-items: end; }}
    .filters label {{ display: grid; gap: 4px; color: var(--fg-muted); font-size: 11px;
                      font-family: var(--font-mono); }}
    .filters input {{ width: 100%; border: 1px solid var(--border); border-radius: var(--radius);
                      background: var(--bg-alt); color: var(--fg); padding: 8px; }}
    .filters button {{ border: 1px solid var(--border); border-radius: var(--radius);
                        background: var(--fg); color: var(--bg); padding: 8px 10px;
                        font-weight: 600; cursor: pointer; }}
    h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em;
          color: var(--fg-muted); margin: 24px 0 12px; }}
  ol.recs {{ list-style: none; padding: 0; margin: 0; display: flex;
            flex-direction: column; gap: 12px; }}
  .rec {{ padding: 16px; border: 1px solid var(--border); border-radius: var(--radius);
          background: var(--bg-alt); }}
  .rec-head {{ display: flex; justify-content: space-between; align-items: baseline;
              margin-bottom: 6px; }}
  .rec-title {{ font-weight: 600; font-family: var(--font-mono); font-size: 13px; }}
  .rec-conf {{ color: var(--fg-muted); font-size: 12px; font-family: var(--font-mono); }}
  .rec-impact {{ color: var(--fg); font-size: 14px; margin-bottom: 8px; }}
  .rec-evidence summary {{ color: var(--accent); cursor: pointer; font-size: 12px;
                          font-family: var(--font-mono); }}
  .rec-evidence[open] summary {{ margin-bottom: 8px; }}
    .evidence-tbl {{ margin-top: 4px; font-size: 12px; }}
    .evidence-tbl th {{ color: var(--fg-muted); font-weight: 500; padding: 4px 10px; }}
    .evidence-tbl td {{ padding: 4px 10px; border-bottom: 1px solid var(--border); }}
    .trace {{ border: 1px solid var(--border); border-radius: var(--radius);
              background: var(--bg-alt); padding: 10px 12px; margin-bottom: 8px; }}
    .trace summary {{ cursor: pointer; font-family: var(--font-mono); font-size: 12px; }}
    .trace[open] summary {{ margin-bottom: 8px; color: var(--accent); }}
    .trace-tbl {{ font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--fg-muted); font-weight: 500; }}
  td.num {{ text-align: right; }}
  .empty {{ padding: 16px; color: var(--fg-muted); font-style: italic; }}
    footer {{ margin-top: 48px; color: var(--fg-muted); font-family: var(--font-mono);
              font-size: 12px; }}
    @media (max-width: 820px) {{
      body {{ padding: 16px; }}
      header {{ display: block; }}
      .filters {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>somm</h1>
  <div class="meta">project: {project} · v{version} · schema v{schema} · {window}d window</div>
</header>

  <section aria-label="System status" role="status" aria-live="polite">
    <div class="status {status_class}">
      <strong>{status_label}</strong> · {hero_line}
    </div>
  </section>

  <form class="filters" action="/" method="get" aria-label="Dashboard filters">
    <label>window days<input name="window" value="{window}" inputmode="numeric"></label>
    <label>search<input name="q" value="{q}" placeholder="session, model, call"></label>
    <label>workload<input name="workload" value="{workload}"></label>
    <label>provider<input name="provider" value="{provider}"></label>
    <label>model<input name="model" value="{model}"></label>
    <button type="submit">Filter</button>
  </form>

  <section aria-label="Recommendations">
    <h2>Top recommendations</h2>
    {recs_html}
  </section>

  <section aria-label="Sessions">
    <h2>Sessions and traces</h2>
    {sessions_html}
  </section>

  <section aria-label="Recent calls">
    <h2>Recent calls</h2>
    {calls_html}
  </section>

  <section aria-label="Evidence">
    <h2>Calls by workload</h2>
  {table_html}
</section>

<footer>
    somm is self-hosted. Binds <code>localhost</code> only by default. Data stays on disk.
    <br>Endpoints: <a href="/health">/health</a> · <a href="/api/status">/api/status</a> · <a href="/api/stats">/api/stats</a> · <a href="/api/calls">/api/calls</a> · <a href="/api/sessions">/api/sessions</a>
  </footer>
</body>
</html>
"""


def _render_table(stats: list[dict]) -> str:
    if not stats:
        return '<div class="empty">No calls yet. Run <code>somm.llm().generate(...)</code> in your Python code.</div>'
    rows = []
    for s in stats:
        rows.append(
            "<tr>"
            f"<td>{_esc(s['workload'])}</td>"
            f"<td>{_esc(s['provider'])}</td>"
            f"<td>{_esc(s['model'])}</td>"
            f"<td class='num'>{s['n_calls']}</td>"
            f"<td class='num'>{s['tokens_in'] or 0}</td>"
            f"<td class='num'>{s['tokens_out'] or 0}</td>"
            f"<td class='num'>{s['n_failed']}</td>"
            f"<td class='num'>{_fmt_table_int(s.get('p95_latency_ms'))}</td>"
            f"<td class='num'>{_fmt_table_int(s.get('p95_ttft_ms'))}</td>"
            f"<td class='num'>{_fmt_table_float(s.get('tpot_ms'))}</td>"
            f"<td class='num'>{_fmt_table_float(s.get('output_tokens_per_second'))}</td>"
            f"<td class='num'>{_fmt_table_pct(s.get('cache_read_ratio'))}</td>"
            f"<td class='num'>{_fmt_table_pct(s.get('goodput_under_slo'))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>workload</th><th>provider</th><th>model</th>"
        "<th class='num'>calls</th><th class='num'>tok in</th>"
        "<th class='num'>tok out</th><th class='num'>fail</th>"
        "<th class='num'>p95 ms</th><th class='num'>ttft p95</th>"
        "<th class='num'>tpot</th><th class='num'>out/s</th>"
        "<th class='num'>cache</th><th class='num'>good</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _fmt_table_int(value: object) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def _fmt_table_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _fmt_table_pct(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.0f}%"


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _parse_positive_int(value: str | None, default: int, *, max_value: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, min(max_value, parsed))


def _status_payload(cfg: Config, repo: Repository, *, window: int) -> dict:
    stats = repo.stats_by_workload(cfg.project, since_days=window)
    total_calls = sum(s["n_calls"] for s in stats)
    total_failed = sum(s["n_failed"] for s in stats)
    total_cost = sum((s.get("cost_usd") or 0.0) for s in stats)
    if total_calls == 0:
        health = "no_data"
    elif total_failed == 0:
        health = "healthy"
    elif (total_failed / total_calls) < 0.2:
        health = "warning"
    else:
        health = "critical"
    return {
        "project": cfg.project,
        "window_days": window,
        "health": health,
        "total_calls": total_calls,
        "total_failed": total_failed,
        "failure_rate": 0.0 if total_calls == 0 else total_failed / total_calls,
        "total_cost_usd": total_cost,
        "active_workloads": len({s["workload"] for s in stats}),
        "load": _load_payload(cfg, repo),
    }


def _load_payload(cfg: Config, repo: Repository) -> dict:
    with repo._open() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS calls_per_minute,
                SUM(CASE WHEN outcome != 'ok' THEN 1 ELSE 0 END) AS failed_per_minute,
                COUNT(DISTINCT workload_id) AS active_workloads,
                COUNT(DISTINCT provider) AS active_providers,
                COUNT(DISTINCT model) AS active_models,
                SUM(tokens_in) AS input_tokens_per_minute,
                SUM(tokens_out) AS output_tokens_per_minute,
                AVG(CASE WHEN outcome = 'ok' THEN latency_ms END) AS mean_latency_ms,
                AVG(CASE WHEN outcome = 'ok' THEN ttft_ms END) AS mean_ttft_ms,
                AVG(
                    CASE
                        WHEN outcome = 'ok'
                         AND ttft_ms IS NOT NULL
                         AND tokens_out > 1
                         AND latency_ms >= ttft_ms
                        THEN ((latency_ms - ttft_ms) * 1.0 / (tokens_out - 1))
                    END
                ) AS mean_tpot_ms
            FROM calls
            WHERE project = ?
              AND ts >= datetime('now', '-60 seconds')
            """,
            (cfg.project,),
        ).fetchone()
    calls = int(row[0] or 0)
    failed = int(row[1] or 0)
    return {
        "window_seconds": 60,
        "calls_per_minute": calls,
        "failed_per_minute": failed,
        "failure_rate": (failed / calls) if calls else 0.0,
        "active_workloads": int(row[2] or 0),
        "active_providers": int(row[3] or 0),
        "active_models": int(row[4] or 0),
        "input_tokens_per_minute": int(row[5] or 0),
        "output_tokens_per_minute": int(row[6] or 0),
        "mean_latency_ms": row[7],
        "mean_ttft_ms": row[8],
        "mean_tpot_ms": row[9],
    }


def _query_calls(
    repo: Repository,
    cfg: Config,
    *,
    window: int,
    limit: int,
    q: str | None = None,
    workload: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> list[dict]:
    clauses = ["c.project = ?", "c.ts >= datetime('now', ?)"]
    params: list[object] = [cfg.project, f"-{window} days"]
    if workload:
        clauses.append("(w.name = ? OR c.workload_id = ?)")
        params.extend([workload, workload])
    if provider:
        clauses.append("c.provider = ?")
        params.append(provider)
    if model:
        clauses.append("c.model = ?")
        params.append(model)
    if q:
        like = f"%{q}%"
        clauses.append(
            "("
            "c.id LIKE ? OR c.provider LIKE ? OR c.model LIKE ? OR "
            "COALESCE(w.name, '') LIKE ? OR COALESCE(c.session_id, '') LIKE ? OR "
            "COALESCE(c.correlation_id, '') LIKE ?"
            ")"
        )
        params.extend([like, like, like, like, like, like])
    where = " AND ".join(clauses)
    params.append(limit)
    with repo._open() as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.ts, COALESCE(w.name, 'ad_hoc') AS workload,
                   c.provider, c.model, c.tokens_in, c.tokens_out,
                   c.latency_ms, c.cost_usd, c.outcome, c.error_kind,
                   c.session_id, c.parent_call_id, c.correlation_id,
                   c.ttft_ms, c.cache_tokens_in, c.cache_tokens_out
            FROM calls c
            LEFT JOIN workloads w ON w.id = c.workload_id
            WHERE {where}
            ORDER BY c.ts DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [
        {
            "id": r[0],
            "ts": r[1],
            "workload": r[2],
            "provider": r[3],
            "model": r[4],
            "tokens_in": r[5],
            "tokens_out": r[6],
            "latency_ms": r[7],
            "cost_usd": r[8] or 0.0,
            "outcome": r[9],
            "error_kind": r[10],
            "session_id": r[11],
            "parent_call_id": r[12],
            "correlation_id": r[13],
            "ttft_ms": r[14],
            "cache_tokens_in": r[15],
            "cache_tokens_out": r[16],
        }
        for r in rows
    ]


def _session_groups(calls: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for call in calls:
        session_id = call.get("session_id") or call["id"]
        group = groups.setdefault(
            session_id,
            {
                "session_id": session_id,
                "first_ts": call["ts"],
                "last_ts": call["ts"],
                "n_calls": 0,
                "n_failed": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "calls": [],
            },
        )
        group["n_calls"] += 1
        group["n_failed"] += 0 if call["outcome"] == "ok" else 1
        group["total_tokens"] += (call["tokens_in"] or 0) + (call["tokens_out"] or 0)
        group["total_cost_usd"] += call["cost_usd"] or 0.0
        group["first_ts"] = min(group["first_ts"], call["ts"])
        group["last_ts"] = max(group["last_ts"], call["ts"])
        group["calls"].append(call)
    for group in groups.values():
        group["calls"].sort(key=lambda c: (c["ts"], c["id"]))
    return sorted(groups.values(), key=lambda g: g["last_ts"], reverse=True)


def _render_calls(calls: list[dict]) -> str:
    if not calls:
        return '<div class="empty">No calls match the current filters.</div>'
    rows = []
    for c in calls[:50]:
        session = c.get("session_id") or ""
        parent = c.get("parent_call_id") or ""
        rows.append(
            "<tr>"
            f"<td><code>{_esc(c['id'][:8])}</code></td>"
            f"<td>{_esc(c['workload'])}</td>"
            f"<td>{_esc(c['provider'])}</td>"
            f"<td>{_esc(c['model'])}</td>"
            f"<td>{_esc(c['outcome'])}</td>"
            f"<td class='num'>{c['latency_ms']}</td>"
            f"<td><code>{_esc(str(session)[:12])}</code></td>"
            f"<td><code>{_esc(str(parent)[:8])}</code></td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>call</th><th>workload</th><th>provider</th><th>model</th>"
        "<th>outcome</th><th class='num'>latency</th><th>session</th><th>parent</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_sessions(groups: list[dict]) -> str:
    if not groups:
        return '<div class="empty">No sessions or traces match the current filters.</div>'
    items = []
    for group in groups[:12]:
        calls = group["calls"]
        rows = []
        for c in calls[:25]:
            parent = c.get("parent_call_id") or ""
            rows.append(
                "<tr>"
                f"<td><code>{_esc(c['id'][:8])}</code></td>"
                f"<td><code>{_esc(str(parent)[:8])}</code></td>"
                f"<td>{_esc(c['workload'])}</td>"
                f"<td>{_esc(c['provider'])}/{_esc(c['model'])}</td>"
                f"<td>{_esc(c['outcome'])}</td>"
                f"<td class='num'>{c['latency_ms']}</td>"
                "</tr>"
            )
        summary = (
            f"{_esc(group['session_id'][:24])} · {group['n_calls']} calls · "
            f"{group['n_failed']} failed · {group['total_tokens']} tokens"
        )
        items.append(
            "<details class='trace'>"
            f"<summary>{summary}</summary>"
            "<table class='trace-tbl'><thead><tr>"
            "<th>call</th><th>parent</th><th>workload</th><th>model</th>"
            "<th>outcome</th><th class='num'>latency</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</details>"
        )
    return "".join(items)


def _list_recommendations(repo: Repository) -> list[dict]:
    """Open (undismissed, unapplied) recommendations, newest first."""
    out = []
    for rec in list_recommendations(repo, open_only=True)[:10]:
        row = rec.as_dict()
        row["confidence"] = row["confidence"] or 0
        out.append(row)
    return out


def _render_recommendations(recs: list[dict]) -> str:
    if not recs:
        return (
            '<div class="empty">No recommendations yet. '
            "Agent runs weekly; needs shadow-eval data or model_intel deltas first.</div>"
        )
    items = []
    for r in recs:
        title = _esc(r["workload"]) + " · " + _esc(r["action"])
        impact = _esc(r["expected_impact"])
        confidence = f"{r['confidence']:.0%}"
        evidence_detail = _render_evidence(r["action"], r["evidence"])
        items.append(
            f'<li class="rec">'
            f'  <div class="rec-head">'
            f'    <span class="rec-title">{title}</span>'
            f'    <span class="rec-conf">confidence {_esc(confidence)}</span>'
            f"  </div>"
            f'  <div class="rec-impact">{impact}</div>'
            f'  <details class="rec-evidence">'
            f"    <summary>evidence</summary>{evidence_detail}"
            f"  </details>"
            f"</li>"
        )
    return f'<ol class="recs" aria-live="polite">{"".join(items)}</ol>'


def _render_evidence(action: str, evidence: dict) -> str:
    if action == "switch_model":
        cur = evidence.get("current", {})
        cand = evidence.get("candidate", {})
        rows = [
            ("", "current", "candidate"),
            ("provider", _esc(cur.get("provider", "")), _esc(cand.get("provider", ""))),
            ("model", _esc(cur.get("model", "")), _esc(cand.get("model", ""))),
            ("quality", _esc(str(cur.get("score", ""))), _esc(str(cand.get("score", "")))),
            ("cost_usd", _esc(str(cur.get("cost_usd", ""))), _esc(str(cand.get("cost_usd", "")))),
            (
                "latency_ms",
                _esc(str(cur.get("latency_ms", ""))),
                _esc(str(cand.get("latency_ms", ""))),
            ),
        ]
        return _evidence_table(rows)
    if action == "new_model_landed":
        cur = evidence.get("current", {})
        cand = evidence.get("candidate", {})
        rows = [
            ("", "current", "candidate"),
            ("provider", _esc(cur.get("provider", "")), _esc(cand.get("provider", ""))),
            ("model", _esc(cur.get("model", "")), _esc(cand.get("model", ""))),
            (
                "in $/1M",
                _esc(str(cur.get("price_in_per_1m", ""))),
                _esc(str(cand.get("price_in_per_1m", ""))),
            ),
            (
                "out $/1M",
                _esc(str(cur.get("price_out_per_1m", ""))),
                _esc(str(cand.get("price_out_per_1m", ""))),
            ),
        ]
        return _evidence_table(rows)
    if action == "chronic_cooldown":
        return (
            "<p>"
            f"provider <code>{_esc(evidence.get('provider', ''))}</code> hit "
            f"circuit-break on {_esc(str(evidence.get('n_calls', '')))} calls. "
            f"{_esc(evidence.get('note', ''))}"
            "</p>"
        )
    return f"<pre>{_esc(json.dumps(evidence, indent=2, sort_keys=True))}</pre>"


def _evidence_table(rows: list[tuple]) -> str:
    head = rows[0]
    body = rows[1:]
    thead = "".join(f"<th>{_esc(c)}</th>" for c in head)
    tbody = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in body)
    return (
        f'<table class="evidence-tbl"><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>'
    )


async def _home(request: Request) -> HTMLResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = _parse_positive_int(request.query_params.get("window"), 7, max_value=365)
    q = request.query_params.get("q") or None
    workload = request.query_params.get("workload") or None
    provider = request.query_params.get("provider") or None
    model = request.query_params.get("model") or None
    stats = repo.stats_by_workload(cfg.project, since_days=window)
    calls = _query_calls(
        repo,
        cfg,
        window=window,
        limit=100,
        q=q,
        workload=workload,
        provider=provider,
        model=model,
    )
    sessions = _session_groups(calls)

    total_calls = sum(s["n_calls"] for s in stats)
    total_failed = sum(s["n_failed"] for s in stats)

    if total_calls == 0:
        status_class, status_label, hero = (
            "",
            "NO DATA YET",
            f"run somm.llm(project={cfg.project!r}).generate(...) to get started",
        )
    elif total_failed == 0:
        status_class, status_label, hero = (
            "",
            "HEALTHY",
            f"{total_calls} calls · 0 failed · {len(stats)} workload(s) active",
        )
    else:
        pct = 100 * total_failed / total_calls
        status_class, status_label, hero = (
            "warn" if pct < 20 else "err",
            "NEEDS ATTENTION",
            f"{total_calls} calls · {total_failed} failed ({pct:.1f}%)",
        )

    try:
        with sqlite3.connect(cfg.db_path) as conn:
            schema_ver = current_schema_version(conn)
    except Exception:
        schema_ver = 0

    recs = _list_recommendations(repo)
    html = _HTML_SHELL.format(
        project=_esc(cfg.project),
        version=_esc(VERSION),
        schema=schema_ver,
        window=window,
        q=_esc(q or ""),
        workload=_esc(workload or ""),
        provider=_esc(provider or ""),
        model=_esc(model or ""),
        status_class=status_class,
        status_label=_esc(status_label),
        hero_line=_esc(hero),
        recs_html=_render_recommendations(recs),
        sessions_html=_render_sessions(sessions),
        calls_html=_render_calls(calls),
        table_html=_render_table(stats),
    )
    return HTMLResponse(html)


async def _health(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    return JSONResponse(
        {
            "ok": True,
            "project": cfg.project,
            "db_exists": cfg.db_path.exists(),
        }
    )


async def _api_stats(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = _parse_positive_int(request.query_params.get("window"), 7, max_value=365)
    stats = repo.stats_by_workload(cfg.project, since_days=window)
    return JSONResponse({"project": cfg.project, "window_days": window, "rows": stats})


async def _api_status(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = _parse_positive_int(request.query_params.get("window"), 7, max_value=365)
    return JSONResponse(_status_payload(cfg, repo, window=window))


async def _api_calls(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = _parse_positive_int(request.query_params.get("window"), 7, max_value=365)
    limit = _parse_positive_int(request.query_params.get("limit"), 100, max_value=1000)
    calls = _query_calls(
        repo,
        cfg,
        window=window,
        limit=limit,
        q=request.query_params.get("q") or None,
        workload=request.query_params.get("workload") or None,
        provider=request.query_params.get("provider") or None,
        model=request.query_params.get("model") or None,
    )
    return JSONResponse(
        {
            "project": cfg.project,
            "window_days": window,
            "count": len(calls),
            "calls": calls,
        }
    )


async def _api_sessions(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = _parse_positive_int(request.query_params.get("window"), 7, max_value=365)
    limit = _parse_positive_int(request.query_params.get("limit"), 500, max_value=2000)
    calls = _query_calls(
        repo,
        cfg,
        window=window,
        limit=limit,
        q=request.query_params.get("q") or None,
        workload=request.query_params.get("workload") or None,
        provider=request.query_params.get("provider") or None,
        model=request.query_params.get("model") or None,
    )
    sessions = _session_groups(calls)
    return JSONResponse(
        {
            "project": cfg.project,
            "window_days": window,
            "count": len(sessions),
            "sessions": sessions,
        }
    )


async def _api_recommendations(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    return JSONResponse({"recommendations": _list_recommendations(repo)})


async def _api_rec_dismiss(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    rec_id = int(request.path_params["rec_id"])
    try:
        dismiss_recommendation(repo, rec_id)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "id": rec_id})


async def _api_rec_apply(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    rec_id = int(request.path_params["rec_id"])
    cfg: Config = request.app.state.config
    mirror_repo = None
    if cfg.cross_project_enabled:
        try:
            mirror_repo = Repository(cfg.global_db_path)
        except Exception:
            mirror_repo = None
    try:
        result = apply_recommendation(
            repo,
            rec_id,
            actor="somm web",
            mirror_repo=mirror_repo,
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse(result.as_dict())


def _otel_value(raw: dict) -> object:
    if "stringValue" in raw:
        return raw["stringValue"]
    if "intValue" in raw:
        try:
            return int(raw["intValue"])
        except (TypeError, ValueError):
            return raw["intValue"]
    if "doubleValue" in raw:
        try:
            return float(raw["doubleValue"])
        except (TypeError, ValueError):
            return raw["doubleValue"]
    if "boolValue" in raw:
        return bool(raw["boolValue"])
    if "arrayValue" in raw:
        values = raw.get("arrayValue", {}).get("values", [])
        return [_otel_value(v) for v in values if isinstance(v, dict)]
    if "kvlistValue" in raw:
        return _otel_attrs(raw.get("kvlistValue", {}).get("values", []))
    return None


def _otel_attrs(items: list | None) -> dict[str, object]:
    attrs: dict[str, object] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and isinstance(value, dict):
            attrs[key] = _otel_value(value)
    return attrs


def _bounded_attr_value(value: object, *, max_chars: int, depth: int = 0) -> object:
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if depth >= 2:
        return str(value)[:max_chars]
    if isinstance(value, list):
        return [
            _bounded_attr_value(item, max_chars=max_chars, depth=depth + 1)
            for item in value[:50]
        ]
    if isinstance(value, dict):
        return {
            _bounded_attr_key(key, max_chars=max_chars): _bounded_attr_value(
                val, max_chars=max_chars, depth=depth + 1
            )
            for key, val in list(value.items())[:50]
        }
    return str(value)[:max_chars]


def _bounded_attr_key(key: object, *, max_chars: int) -> str:
    # Keep normal semantic-convention keys intact even under a tiny value cap,
    # while still bounding pathological key names.
    return str(key)[: max(64, max_chars)]


def _bound_attrs(attrs: dict[str, object], *, max_chars: int) -> dict[str, object]:
    max_chars = max(1, int(max_chars))
    return {
        _bounded_attr_key(key, max_chars=max_chars): _bounded_attr_value(
            value, max_chars=max_chars
        )
        for key, value in attrs.items()
    }


def _iter_otlp_spans(payload: dict, *, max_attr_chars: int):
    for resource_span in payload.get("resourceSpans", []) or []:
        if not isinstance(resource_span, dict):
            continue
        resource_attrs = _otel_attrs(
            (resource_span.get("resource") or {}).get("attributes")
        )
        span_groups = []
        span_groups.extend(resource_span.get("scopeSpans", []) or [])
        span_groups.extend(resource_span.get("instrumentationLibrarySpans", []) or [])
        for span_group in span_groups:
            if not isinstance(span_group, dict):
                continue
            scope = span_group.get("scope") or span_group.get("instrumentationLibrary") or {}
            scope_attrs = _otel_attrs(scope.get("attributes"))
            for span in span_group.get("spans", []) or []:
                if not isinstance(span, dict):
                    continue
                attrs = dict(resource_attrs)
                attrs.update(scope_attrs)
                attrs.update(_otel_attrs(span.get("attributes")))
                yield span, _bound_attrs(attrs, max_chars=max_attr_chars)
    for span in payload.get("spans", []) or []:
        if isinstance(span, dict):
            yield span, _bound_attrs(_otel_attrs(span.get("attributes")), max_chars=max_attr_chars)


def _attr(attrs: dict[str, object], *keys: str, default=None):
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_time(value: object) -> datetime:
    nanos = _as_int(value, 0)
    if nanos <= 0:
        return datetime.now(UTC)
    return datetime.fromtimestamp(nanos / 1_000_000_000, UTC)


def _latency_ms(span: dict, attrs: dict[str, object]) -> int:
    explicit = _attr(attrs, "somm.latency_ms", "gen_ai.latency_ms", "llm.latency_ms")
    if explicit is not None:
        return max(0, _as_int(explicit, 0))
    start = _as_int(span.get("startTimeUnixNano"), 0)
    end = _as_int(span.get("endTimeUnixNano"), 0)
    if start > 0 and end >= start:
        return int((end - start) / 1_000_000)
    return 0


def _call_id_for_span(trace_id: str, span_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"somm-otlp:{trace_id}:{span_id}"))


def _outcome_for_span(span: dict) -> tuple[Outcome, str | None]:
    status = span.get("status") or {}
    code = status.get("code") if isinstance(status, dict) else None
    if code in ("STATUS_CODE_ERROR", "ERROR", 2):
        return Outcome.ERROR, str(status.get("message") or "otel_status_error")
    return Outcome.OK, None


def _call_from_otlp_span(
    repo: Repository,
    cfg: Config,
    span: dict,
    attrs: dict[str, object],
) -> Call:
    trace_id = str(span.get("traceId") or _attr(attrs, "trace_id", default=""))
    span_id = str(span.get("spanId") or _attr(attrs, "span_id", default=""))
    if not trace_id:
        trace_id = stable_hash({"span": span, "attrs": attrs})
    if not span_id:
        span_id = stable_hash({"span": span.get("name"), "attrs": attrs})
    provider = str(
        _attr(attrs, "somm.provider", "gen_ai.system", "llm.provider", default="otel")
    )
    model = str(
        _attr(
            attrs,
            "somm.model",
            "gen_ai.response.model",
            "gen_ai.request.model",
            "llm.model_name",
            default=span.get("name") or "unknown",
        )
    )
    workload = str(
        _attr(
            attrs,
            "somm.workload",
            "llm.workload",
            "gen_ai.operation.name",
            default=span.get("name") or "otel",
        )
    )
    workload_id = repo.register_workload(name=workload, project=cfg.project).id
    outcome, error_kind = _outcome_for_span(span)
    parent_span_id = str(span.get("parentSpanId") or "")
    parent_call_id = _call_id_for_span(trace_id, parent_span_id) if parent_span_id else None
    prompt_hash = stable_hash({"trace_id": trace_id, "span_id": span_id, "side": "prompt"})
    response_hash = stable_hash({"trace_id": trace_id, "span_id": span_id, "side": "response"})
    return Call(
        id=_call_id_for_span(trace_id, span_id),
        ts=_span_time(span.get("startTimeUnixNano")),
        project=cfg.project,
        workload_id=workload_id,
        prompt_id=None,
        provider=provider,
        model=model,
        tokens_in=_as_int(
            _attr(
                attrs,
                "somm.tokens_in",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.prompt_tokens",
                "llm.usage.prompt_tokens",
            )
        ),
        tokens_out=_as_int(
            _attr(
                attrs,
                "somm.tokens_out",
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.completion_tokens",
                "llm.usage.completion_tokens",
            )
        ),
        latency_ms=_latency_ms(span, attrs),
        cost_usd=_as_float(_attr(attrs, "somm.cost_usd", "gen_ai.usage.cost_usd")),
        outcome=outcome,
        error_kind=error_kind,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
        correlation_id=trace_id,
        ttft_ms=_as_int(_attr(attrs, "somm.ttft_ms", "gen_ai.ttft_ms"), 0) or None,
        session_id=str(_attr(attrs, "somm.session_id", "session.id", default=trace_id)),
        parent_call_id=parent_call_id,
        cache_tokens_in=_as_int(
            _attr(attrs, "somm.cache_tokens_in", "gen_ai.usage.cache_read_input_tokens"),
            0,
        ) or None,
        cache_tokens_out=_as_int(_attr(attrs, "somm.cache_tokens_out"), 0) or None,
    )


def _ingest_otlp_payload(
    repo: Repository,
    cfg: Config,
    payload: dict,
    *,
    max_spans: int,
    max_attr_chars: int,
) -> tuple[dict, int]:
    spans = []
    for idx, (span, attrs) in enumerate(
        _iter_otlp_spans(payload, max_attr_chars=max_attr_chars),
        start=1,
    ):
        if idx > max_spans:
            return (
                {
                    "ok": False,
                    "error": f"OTLP payload exceeds {max_spans} spans",
                    "max_spans": max_spans,
                },
                413,
            )
        spans.append((span, attrs))

    ingested = 0
    duplicates = 0
    skipped = 0
    for span, attrs in spans:
        try:
            call = _call_from_otlp_span(repo, cfg, span, attrs)
            repo.write_call(call)
            ingested += 1
        except sqlite3.IntegrityError:
            duplicates += 1
        except Exception:
            skipped += 1
    return {
        "ok": True,
        "ingested": ingested,
        "duplicates": duplicates,
        "skipped": skipped,
    }, 200


async def _api_otlp_traces(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    try:
        payload = await read_bounded_json(request, max_bytes=cfg.service_otlp_max_body_bytes)
    except PayloadTooLarge as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=413)
    except json.JSONDecodeError:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "error": "OTLP payload must be an object"}, status_code=400)
    result, status = _ingest_otlp_payload(
        repo,
        cfg,
        payload,
        max_spans=cfg.service_otlp_max_spans,
        max_attr_chars=cfg.service_otlp_max_attr_chars,
    )
    return JSONResponse(result, status_code=status)


async def _api_version(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    try:
        with sqlite3.connect(cfg.db_path) as conn:
            schema_ver = current_schema_version(conn)
    except Exception:
        schema_ver = 0
    return JSONResponse(
        {
            "version": VERSION,
            "schema_version": schema_ver,
            "project": cfg.project,
        }
    )


def create_app(config: Config | None = None) -> Starlette:
    cfg = config or load_config()
    repo = Repository(cfg.db_path)
    service_token = load_service_token(cfg)
    app = Starlette(
        debug=False,
        routes=[
            Route("/", _home),
            Route("/health", _health),
            Route("/api/status", _api_status),
            Route("/api/stats", _api_stats),
            Route("/api/calls", _api_calls),
            Route("/api/sessions", _api_sessions),
            Route("/api/version", _api_version),
            Route("/api/recommendations", _api_recommendations),
            Route("/api/recommendations/{rec_id:int}/dismiss", _api_rec_dismiss, methods=["POST"]),
            Route("/api/recommendations/{rec_id:int}/apply", _api_rec_apply, methods=["POST"]),
            Route("/api/otlp/v1/traces", _api_otlp_traces, methods=["POST"]),
            Route("/v1/traces", _api_otlp_traces, methods=["POST"]),
            Route("/v1/messages", messages_endpoint, methods=["POST"]),
        ],
    )
    app.add_middleware(
        LocalSecurityMiddleware,
        token=service_token.value,
        public_read=cfg.service_public_read,
    )
    app.state.config = cfg
    app.state.repo = repo
    app.state.service_token = service_token
    return app


# Moved to somm_service.inprocess (workers-only import weight, no
# starlette) — re-exported here for backward compatibility.
from somm_service.inprocess import (  # noqa: E402
    build_workers_factory as _build_workers_factory,
)
from somm_service.inprocess import (  # noqa: E402,F401
    start_inprocess_scheduler,
)


def run_server(
    config: Config | None = None,
    host: str = "127.0.0.1",
    port: int = 7878,
    log_level: str = "info",
    enable_scheduler: bool = True,
) -> None:
    """Run the web admin + API server. Starts the scheduler unless disabled."""
    import uvicorn

    app = create_app(config)
    cfg: Config = app.state.config
    repo: Repository = app.state.repo
    _log_service_token(app.state.service_token, host=host, port=port)

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "\n⚠️  somm serve is binding to a non-localhost address.\n"
            "   Trace data stays in plain SQLite files on disk.\n"
            "   Only do this if you know what you're doing.\n"
        )

    scheduler = None
    if enable_scheduler:
        from somm_service.workers import Scheduler

        scheduler = Scheduler(repo, _build_workers_factory(cfg, repo))
        scheduler.start()
        app.state.scheduler = scheduler

    try:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    finally:
        if scheduler is not None:
            scheduler.stop()
