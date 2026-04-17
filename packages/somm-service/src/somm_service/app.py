"""Starlette app serving the web admin + HTTP API.

D1 surface:
  GET /                       HTML dashboard — status line + rec placeholder
  GET /health                 JSON liveness probe
  GET /api/stats              JSON roll-up (per-workload × provider × model)
  GET /api/version            JSON service + schema version

Design tokens + a11y spec from PLAN.md Phase 2 applied inline (v0.1 ships
tokens in-HTML; `packages/somm-service/web/tokens.css` lands in D4).
"""

from __future__ import annotations

import sqlite3

from somm_core import VERSION
from somm_core.config import Config
from somm_core.config import load as load_config
from somm_core.repository import Repository
from somm_core.schema import current_schema_version
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

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
  <div class="empty">No recommendations yet — agent runs weekly and needs more telemetry first.</div>
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


_ESC_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;"}


def _esc(s: str) -> str:
    return "".join(_ESC_MAP.get(c, c) for c in str(s))


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

    html = _HTML_SHELL.format(
        project=_esc(cfg.project),
        version=_esc(VERSION),
        schema=schema_ver,
        window=window,
        status_class=status_class,
        status_label=_esc(status_label),
        hero_line=_esc(hero),
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
    app = Starlette(
        debug=False,
        routes=[
            Route("/", _home),
            Route("/health", _health),
            Route("/api/stats", _api_stats),
            Route("/api/version", _api_version),
        ],
    )
    app.state.config = cfg
    app.state.repo = repo
    return app


def run_server(
    config: Config | None = None,
    host: str = "127.0.0.1",
    port: int = 7878,
    log_level: str = "info",
) -> None:
    """Run the web admin + API server. Blocks until Ctrl-C."""
    import uvicorn

    app = create_app(config)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "\n⚠️  somm serve is binding to a non-localhost address.\n"
            "   Trace data stays in plain SQLite files on disk.\n"
            "   Only do this if you know what you're doing.\n"
        )
    uvicorn.run(app, host=host, port=port, log_level=log_level)
