"""Starlette app serving the web admin + HTTP API.

HTTP surface:
  GET /                        HTML dashboard — status line + recs + stats
  GET /health                  JSON liveness probe
  GET /api/stats               JSON roll-up (per-workload × provider × model)
  GET /api/version             JSON service + schema version
  GET /api/recommendations     JSON open recs
  POST /api/recommendations/{id}/dismiss
  POST /api/recommendations/{id}/apply

Design tokens + a11y spec from PLAN.md Phase 2 applied inline (v0.1 ships
tokens in-HTML; `packages/somm-service/web/tokens.css` lands when we extract).

`somm serve` also starts a Scheduler background thread that runs the
model_intel / shadow_eval / agent workers on their cadences.
"""

from __future__ import annotations

import json
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


_ESC_MAP = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;"}


def _esc(s: str) -> str:
    return "".join(_ESC_MAP.get(c, c) for c in str(s))


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
        ],
    )
    app.state.config = cfg
    app.state.repo = repo
    return app


def _build_workers_factory(cfg: Config, repo: Repository):
    """Create a factory that returns a worker instance for a given job name."""
    from somm.providers.anthropic import AnthropicProvider
    from somm.providers.minimax import MinimaxProvider
    from somm.providers.ollama import OllamaProvider
    from somm.providers.openai import OpenAIProvider
    from somm.providers.openrouter import OpenRouterProvider

    from somm_service.workers import (
        AgentWorker,
        ModelIntelWorker,
        ShadowEvalWorker,
    )

    def _providers_for_shadow() -> list:
        chain = [OllamaProvider(base_url=cfg.ollama_url, default_model=cfg.ollama_model)]
        if cfg.openrouter_api_key:
            chain.append(
                OpenRouterProvider(
                    api_key=cfg.openrouter_api_key,
                    roster=cfg.openrouter_roster,
                )
            )
        if cfg.minimax_api_key:
            chain.append(
                MinimaxProvider(
                    api_key=cfg.minimax_api_key,
                    default_model=cfg.minimax_model,
                )
            )
        if cfg.anthropic_api_key:
            chain.append(
                AnthropicProvider(
                    api_key=cfg.anthropic_api_key,
                    default_model=cfg.anthropic_model,
                )
            )
        if cfg.openai_api_key:
            chain.append(
                OpenAIProvider(
                    api_key=cfg.openai_api_key,
                    base_url=cfg.openai_base_url,
                    default_model=cfg.openai_model,
                )
            )
        return chain

    def factory(job_name: str):
        if job_name == "model_intel":
            return ModelIntelWorker(repo, ollama_url=cfg.ollama_url)
        if job_name == "shadow_eval":
            return ShadowEvalWorker(repo, providers=_providers_for_shadow())
        if job_name == "agent":
            return AgentWorker(repo)
        return None

    return factory


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
