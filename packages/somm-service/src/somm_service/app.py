"""Starlette app serving the web admin + HTTP API.

HTTP surface:
  GET /                        HTML dashboard — status line + recs + stats
  GET /health                  JSON liveness probe
  GET /api/stats               JSON roll-up (per-workload × provider × model)
  GET /api/version             JSON service + schema version
  GET /api/recommendations     JSON open recs
  POST /api/recommendations/{id}/dismiss
  POST /api/recommendations/{id}/apply
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
from pathlib import Path
from urllib.parse import urlsplit

from somm_core import VERSION
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.repository import Repository
from somm_core.schema import current_schema_version
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from somm_service.proxy import _anthropic_error, messages_endpoint

_CSP = "default-src 'none'; style-src 'unsafe-inline'"
_LOCAL_HEADER = "x-somm-local"
_TOKEN_ENV_VAR = "SOMM_SERVICE_TOKEN"


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
            f"{token.value} "
            f"(use: curl -H 'Authorization: Bearer {token.value}' http://{host}:{port}/v1/messages)"
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
    def __init__(self, app, *, token: str) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        method = scope["method"]
        path = scope["path"]
        protected_admin = method == "POST" and path.startswith("/api/recommendations/")
        protected_messages = method == "POST" and path == "/v1/messages"

        if protected_admin or protected_messages:
            if not self._is_authorized(headers):
                response = self._forbidden(protected_messages)
                self._set_security_headers(response)
                await response(scope, receive, send)
                return
            if protected_messages and not self._is_json_request(headers):
                response = _anthropic_error(
                    error_type="invalid_request_error",
                    message="POST /v1/messages requires Content-Type: application/json",
                    status=415,
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
  table {{ width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--fg-muted); font-weight: 500; }}
  td.num {{ text-align: right; }}
  .empty {{ padding: 16px; color: var(--fg-muted); font-style: italic; }}
  footer {{ margin-top: 48px; color: var(--fg-muted); font-family: var(--font-mono);
            font-size: 12px; }}
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

<section aria-label="Recommendations">
  <h2>Top recommendations</h2>
  {recs_html}
</section>

<section aria-label="Evidence">
  <h2>Calls by workload</h2>
  {table_html}
</section>

<footer>
  somm is self-hosted. Binds <code>localhost</code> only by default. Data stays on disk.
  <br>Endpoints: <a href="/health">/health</a> · <a href="/api/stats">/api/stats</a> · <a href="/api/version">/api/version</a>
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
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>workload</th><th>provider</th><th>model</th>"
        "<th class='num'>calls</th><th class='num'>tok in</th>"
        "<th class='num'>tok out</th><th class='num'>fail</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _list_recommendations(repo: Repository) -> list[dict]:
    """Open (undismissed, unapplied) recommendations, newest first."""
    with repo._open() as conn:
        rows = conn.execute(
            "SELECT r.id, r.workload_id, w.name, r.action, r.evidence_json, "
            "       r.expected_impact, r.confidence, r.created_at "
            "FROM recommendations r "
            "LEFT JOIN workloads w ON w.id = r.workload_id "
            "WHERE r.dismissed_at IS NULL AND r.applied_at IS NULL "
            "ORDER BY r.created_at DESC LIMIT 10"
        ).fetchall()
    out = []
    for r in rows:
        try:
            evidence = json.loads(r[4]) if r[4] else {}
        except json.JSONDecodeError:
            evidence = {}
        out.append(
            {
                "id": r[0],
                "workload_id": r[1],
                "workload": r[2] or "(unknown)",
                "action": r[3],
                "evidence": evidence,
                "expected_impact": r[5] or "",
                "confidence": r[6] or 0,
                "created_at": r[7],
            }
        )
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
    window = int(request.query_params.get("window", "7"))
    stats = repo.stats_by_workload(cfg.project, since_days=window)

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
        status_class=status_class,
        status_label=_esc(status_label),
        hero_line=_esc(hero),
        recs_html=_render_recommendations(recs),
        table_html=_render_table(stats),
    )
    return HTMLResponse(html)


async def _health(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    return JSONResponse(
        {
            "ok": True,
            "project": cfg.project,
            "db_path": str(cfg.db_path),
            "db_exists": cfg.db_path.exists(),
        }
    )


async def _api_stats(request: Request) -> JSONResponse:
    cfg: Config = request.app.state.config
    repo: Repository = request.app.state.repo
    window = int(request.query_params.get("window", "7"))
    stats = repo.stats_by_workload(cfg.project, since_days=window)
    return JSONResponse({"project": cfg.project, "window_days": window, "rows": stats})


async def _api_recommendations(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    return JSONResponse({"recommendations": _list_recommendations(repo)})


async def _api_rec_dismiss(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    rec_id = int(request.path_params["rec_id"])
    with repo._open() as conn:
        conn.execute(
            "UPDATE recommendations SET dismissed_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND dismissed_at IS NULL",
            (rec_id,),
        )
    return JSONResponse({"ok": True, "id": rec_id})


async def _api_rec_apply(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repo
    rec_id = int(request.path_params["rec_id"])
    with repo._open() as conn:
        conn.execute(
            "UPDATE recommendations SET applied_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND applied_at IS NULL",
            (rec_id,),
        )
    return JSONResponse({"ok": True, "id": rec_id})


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
            Route("/api/stats", _api_stats),
            Route("/api/version", _api_version),
            Route("/api/recommendations", _api_recommendations),
            Route("/api/recommendations/{rec_id:int}/dismiss", _api_rec_dismiss, methods=["POST"]),
            Route("/api/recommendations/{rec_id:int}/apply", _api_rec_apply, methods=["POST"]),
            Route("/v1/messages", messages_endpoint, methods=["POST"]),
        ],
    )
    app.add_middleware(LocalSecurityMiddleware, token=service_token.value)
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
