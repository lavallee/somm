# Contributing to somm

somm is self-hosted LLM telemetry, routing, and an intelligence loop —
built for individual developers and small teams who want visibility
and cost control over their LLM usage without phoning home to anyone.
Contributions welcome.

## Dev setup

```bash
git clone https://github.com/lavallee/somm && cd somm
uv sync --all-packages
uv run pytest packages/ tests/
uv run ruff check packages/ tests/
```

That's it — no separate venv bootstrapping, no Docker required.

## Monorepo layout

- `packages/somm` — the library: client, routing, sommelier, CLI (`somm`)
- `packages/somm-core` — shared core: schema, repository, config, parse, version
- `packages/somm-service` — web admin, HTTP API, background workers (`somm-serve`)
- `packages/somm-mcp` — MCP server exposing telemetry + recommendations to coding agents (`somm-mcp`)
- `packages/somm-langchain` — LangChain `BaseChatModel` adapter
- `packages/somm-skill` — onboarding skill templates for coding agents

## Making a change

1. Write the code and tests. Behavior changes need tests — a PR that
   changes what somm does without a test covering it will get bounced.
2. Add a `CHANGELOG.md` entry under `## [Unreleased]`. Be specific
   about what changed and why; future-you and downstream users both
   read this file.
3. If your change touches versioned files (any `pyproject.toml`,
   `packages/somm-core/src/somm_core/version.py`), see
   [`RELEASING.md`](./RELEASING.md) — all package versions move in
   lockstep, never independently.
4. Run the full check before opening a PR:
   ```bash
   uv run pytest packages/ tests/
   uv run ruff check packages/ tests/
   ```

## The blocklist guard

`tests/test_blocklist.py` scans tracked files for strings that would
leak the author's private project names or local filesystem paths
into the open-source repo. It must pass on every PR — if it fails,
you've likely pasted output from a local run that included a personal
path or an unrelated internal project name. Sanitize the string
(e.g. replace `/home/<you>/...` with `/home/user/...`) rather than
adding an exemption.

## Live-ollama tests

A few tests talk to a local Ollama instance and auto-skip when one
isn't running (`no local ollama`). That's expected and not a failure —
CI runs without Ollama installed, so you'll see those skips there too.
Any *other* skip or failure blocks merge.
