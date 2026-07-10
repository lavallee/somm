# somm Threat Model

somm is a local-first LLM telemetry and routing system. It does not run a
somm-operated cloud service and does not phone home. The primary trust boundary
is the user's machine: provider calls leave the host only because the user
configured a provider key or local provider endpoint.

## Assets

- Provider API keys and CLI credentials.
- Prompt and response bodies when sample capture is enabled.
- Local telemetry databases under `.somm/` and `~/.somm/`.
- Model decisions mirrored to the global local store.
- Workload policies, budgets, and recommendation actions.

## Trust Boundaries

- **Library hot path**: user code imports `somm` and calls providers. Hook and
  plugin failures must never expose secrets or break an otherwise successful
  call.
- **Web service**: `somm serve` binds to localhost by default. Dashboard/read
  APIs, mutating routes, and LLM proxy routes require the service bearer token
  or the same-origin local dashboard header path, unless read APIs are
  explicitly opened with `SOMM_SERVICE_PUBLIC_READ=1`.
- **MCP stdio**: `somm-mcp` talks to a local coding agent over stdio. MCP tool
  results wrap stored prompt/response bodies in untrusted-content envelopes.
- **OTLP ingest**: `/api/otlp/v1/traces` and `/v1/traces` accept JSON spans from
  local/polyglot apps. They are write endpoints and require the service token.
- **Provider egress**: configured providers receive prompts and tool schemas.
  Private workloads are blocked from online evaluation egress.

## In-Scope Threats

- Localhost CSRF and DNS rebinding against the dashboard or proxy.
- Prompt injection inside stored samples returned through MCP.
- Accidental trace/body disclosure through web views, JSON APIs, plugins, or
  logs.
- Over-broad filesystem permissions on local SQLite databases and service token
  files.
- Malformed OTLP, spool, or JSON-extraction input causing crashes or unbounded
  writes.
- GitHub Actions supply-chain drift in dependency and workflow configuration.

## Mitigations

- Web admin binds `127.0.0.1` by default and warns on non-localhost bind.
- Dashboard/read APIs and mutating service routes require bearer-token auth;
  dashboard-only header auth is accepted only for loopback hosts with
  same-origin evidence.
- Public `/health` reports liveness without local filesystem paths. Richer
  status, calls, sessions, recommendation, and version data use the read API
  auth boundary by default.
- HTML is manually escaped and served with `default-src 'none'` plus nosniff and
  no-referrer headers.
- Local DB directories are created `0700`; SQLite files are chmod `0600`.
- MCP responses envelope stored user/model content as untrusted text.
- Private workloads cannot be shadow-graded; the SQL `shadow_candidates` view
  and worker checks both enforce this.
- Service OTLP ingest is lenient but bounded to normal call rows; oversized
  JSON bodies and over-cap span batches are rejected before writes, attributes
  are truncated, and malformed spans are skipped rather than failing the whole
  batch.
- The local proxy bounds request bodies, runs provider dispatch off the event
  loop with service-controlled timeouts, and rejects unknown explicit proxy
  workloads so authenticated clients cannot bypass pre-registered caps.
- `somm_compare` caps model fanout and per-call `max_tokens`; callers must set
  `allow_expensive=true` for the elevated cap, and that elevated cap remains a
  hard ceiling.
- CI runs `pip-audit` against the locked dependency export and `zizmor` against
  GitHub Actions workflows.

## Accepted Risks

- A process with arbitrary code execution as the same OS user can read local
  `.somm` files and environment variables. This is out of scope for somm.
- If the user explicitly binds `somm serve` to a network interface, they are
  responsible for network placement and any reverse-proxy authentication.
- Installed provider or hook plugins are trusted local code. Install only
  plugins you would trust to run in the same Python process as your app.

## Reporting

Report vulnerabilities privately using the process in `SECURITY.md`. Include
the somm version, deployment mode, whether sample capture was enabled, and a
minimal reproduction when possible.
