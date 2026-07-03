# Security Policy

## Supported versions

Only the latest published release of somm is supported with security
fixes. There is no long-term-support branch — please upgrade before
reporting an issue against an older version.

## Reporting a vulnerability

Please report security issues privately by emailing
**mlavallee@gmail.com** rather than opening a public GitHub issue.
Include:

- A description of the issue and its potential impact
- Steps to reproduce (a minimal repro is ideal)
- The somm version (`somm doctor` includes this) and platform

You should get an acknowledgement within a few days. Once a fix is
ready, we'll coordinate disclosure and credit with you.

## Scope

somm is local-first and self-hosted: there is no somm-operated
service, no telemetry phone-home, and no central account. The
relevant attack surfaces are:

- **The localhost web admin** (`somm-serve`) — binds to a local
  interface for reviewing telemetry and recommendations. Don't expose
  it to an untrusted network without your own auth/reverse-proxy layer.
- **The MCP stdio server** (`somm-mcp`) — talks to coding agents over
  stdio; treat it like any other local tool with filesystem/process
  access.
- **Provider API-key handling** — somm holds API keys for the LLM
  providers you configure (OpenAI, Anthropic, OpenRouter, etc.) in
  order to route calls on your behalf. Key material should never be
  logged, written to telemetry records, or transmitted anywhere other
  than the configured provider endpoint.
- **Local SQLite storage** — somm's telemetry and config databases are
  created with `chmod 0600` so only the owning user can read them.
  Report any code path that creates these files with broader
  permissions.

Out of scope: vulnerabilities that require an attacker to already have
arbitrary code execution on the machine running somm, or that require
physical access to the host.
