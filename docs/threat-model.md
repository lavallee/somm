# somm Threat Model

somm is a local-first LLM telemetry and routing system. It does not run a
somm-operated cloud service and does not phone home. The primary trust boundary
is the user's machine: provider calls leave the host only because the user
configured a provider key or local provider endpoint.

## Assets

- Provider API keys and CLI credentials.
- Prompt and response bodies, and the request parameters sent with them
  (system prompt, tool definitions), when sample capture is enabled.
- Local telemetry databases under `.somm/` and `~/.somm/`.
- Model decisions mirrored to the global local store.
- Workload policies, budgets, and recommendation actions.

## Trust Boundaries

- **Library hot path**: user code imports `somm` and calls providers. Hook and
  plugin failures must never expose secrets or break an otherwise successful
  call.
- **Web service**: `somm serve` binds to localhost by default. The LLM proxy
  routes, `/api/routing/decide` and OTLP ingest always require the service
  bearer token. Dashboard reads and recommendation actions accept the token or
  the bundled dashboard's local header path; read APIs can be opened
  deliberately with `SOMM_SERVICE_PUBLIC_READ=1`.
- **MCP stdio**: `somm-mcp` talks to a local coding agent over stdio. MCP tool
  results wrap stored prompt/response bodies in untrusted-content envelopes.
- **OTLP ingest**: `/api/otlp/v1/traces` and `/v1/traces` accept spans from
  local/polyglot apps as JSON, or as protobuf with the `somm-service[otlp]`
  extra. They are write endpoints and require the service token.
- **Provider egress**: configured providers receive prompts and tool schemas.
  Each route is classified into a data tier (`local`, `zdr`, `subscribed`,
  `unvetted`); a workload's data boundary limits which routes it may reach
  (`docs/data-boundary.md`). Private workloads are bounded to `local` and are
  blocked from online evaluation egress.

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
- Service routes require bearer-token auth. The dashboard's header path
  (`X-Somm-Local: 1`) is accepted only on dashboard reads and recommendation
  actions, only from a loopback peer address with a loopback `Host` and
  same-origin evidence, and never when a forwarding header is present (a
  reverse proxy or tunnel makes every request arrive from loopback). The
  headers are chosen by the client; the peer address is not.
- Public `/health` reports liveness without local filesystem paths. Richer
  status, calls, sessions, recommendation, and version data use the read API
  auth boundary by default.
- HTML is manually escaped and served with `default-src 'none'` plus nosniff and
  no-referrer headers.
- Local DB directories are created `0700`; SQLite files are chmod `0600`.
- MCP responses envelope stored user/model content as untrusted text.
- Private workloads cannot be shadow-graded; the SQL `shadow_candidates` view
  and the shadow-eval worker both enforce this, and `enable_shadow` refuses.
- Data boundary: one `admit()` check before every egress (router, pinned calls,
  streams, OpenRouter roster walk, workload handlers, shadow gold and judge
  routes, the proxy routes, MCP compare, harness attempts). Unclassifiable
  routes are `unvetted`; an unreadable `~/.somm/boundary.toml` loosens
  nothing. Refusals are not retried, never fall through to a looser route, and
  are recorded as ledger rows. A tier is a declaration about the owner's
  contracts, not a proof of what the far end retains; `pre_call` hooks and
  provider plugins run before or outside the check and are trusted code.
- Service OTLP ingest is lenient but bounded to normal call rows; oversized
  JSON bodies and over-cap span batches are rejected before writes, attributes
  are truncated, and malformed spans are skipped rather than failing the whole
  batch.
- The local proxy bounds request bodies, runs provider dispatch off the event
  loop through `SommLLM` (provider timeouts, boundary and budget gates
  included), refuses wire fields it cannot honor, and rejects unknown explicit
  proxy workloads so authenticated clients cannot bypass pre-registered caps.
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
