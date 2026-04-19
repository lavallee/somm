# somm — TODOs (deferred scope)

_Items deferred during autoplan review to keep v0.1 focused. Tracked for post-v0.1 consideration._

## Deferred (in priority order)

### Core product
- **A/B routing** — agent recommendations become live shadow traffic splits with lift calculation. Currently agent only recommends; no closed loop. (~2–3d CC.)
- **`somm.ensemble(prompt, models=[…], aggregate=fn)`** — parallel-model call primitive for ensembling. (~2–3d CC.)
- **Auto-eval generation from production samples** — frontier writes grading rubrics from sampled call pairs; builds eval suites automatically. (~2d CC.)

### Infrastructure
- **Postgres backend** for small-team shared deployments as an optional `somm[postgres]` extra. SQLite remains default. (Phase 3 codex.)
- **Windows service lifecycle** support (Task Scheduler integration). Linux + macOS day-one. (Phase 3.)
- **HF trending model-intel source** — behind feature flag; OpenRouter is primary source. Fragile DOM scraping. (Phase 3 H1.)
- **Release-feed model-intel sources** (RSS/Atom per-provider) — most are dead; feature-flagged. (Phase 3 H1.)
- **Provider-specific tokenizers** as `somm[tokenizers]` extras (tiktoken, etc.). Default approximation (4 chars/token) ships in v0.1. (Phase 3 H2.)
- **Broader model-intel signal sources for sommelier ranking** — see [`docs/intel-sources.md`](./docs/intel-sources.md). Candidates: LMArena Elo (quality), Artificial Analysis (price/speed/quality), canirun.ai (local GPU feasibility), LiveBench (contamination-resistant), Open LLM Leaderboard (per-benchmark). Each needs: stable source URL, refresh cadence, failure mode when the source is down. Likely feature-flagged per-source like the HF scraper.

### DX
- **Beacon telemetry** for DX measurement — v0.1 is local-only `somm admin dx-report`; beacon (opt-in) deferred. (Phase 3.5.)
- **`somm plugin` command** — install/list/remove plugins (providers, graders, etc.) with supply-chain checks. Currently pip-based. (Phase 3.5 extension.)
- **GUI installer / macOS .dmg / Linux .deb** — v0.1 ships pipx/uv tool install; packaged installers post-v0.1.

### Design
- **Recommendation evidence detail panel design review** — v0.1 spec says inline in card; deep drawer/modal design deferred.
- **Dashboard filtering/search** — v0.1 has per-project toggle + time window dropdown; richer filtering deferred.
- **Dark-mode polish** — tokens.css has `prefers-color-scheme` media query; light-mode a second-class citizen for v0.1.

## Principles for pulling items back into scope

- If a deferred item becomes load-bearing for a post-v0.1 user demand, promote.
- If a deferred item can be shipped as an optional extra (`somm[X]`) without bloating core, that path is preferred over blocking v0.1.
- If a deferred item would be a days-of-work surprise to someone trying to build it themselves (plugin protocol, extensibility), promote to earlier.
