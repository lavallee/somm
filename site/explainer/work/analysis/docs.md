## Purpose

This area explains somm’s architecture, operating model, public workflows, extension points, security posture, and canonical errors. It presents the system as a local, per-project SQLite telemetry substrate on which routing, evaluation, model intelligence, recommendations, and agent interfaces are layered (`docs/BLUEPRINT.md:19`).

## How it works

Each `SommLLM.generate()` call resolves workload policy, runs mutable `pre_call` hooks, routes across capable providers with fallback, records an immutable `calls` row, then runs inline and background observers (`docs/plugins.md:9`). Workloads carry privacy, budget, and capability requirements; calls, updates, optional samples, evaluations, model intelligence, health state, and recommendations share one SQLite data model (`docs/BLUEPRINT.md:118`). Services consume calls to produce evaluations and recommendations without rewriting the original telemetry (`docs/BLUEPRINT.md:171`).

Routing filters explicitly incapable provider/model pairs before dispatch, while unknown capability values remain eligible (`docs/multimodal.md:75`). Transient failures such as timeouts, rate limits, 5xx responses, and insufficient credit cool the provider and fall through; fatal policy or configuration failures stop immediately. Canonical `SOMM_*` pages document each problem, cause, behavior, and fix (`docs/errors/index.html:28`).

Model intelligence combines bundled and refreshed pricing/capability data with optional sources; internal shadow evaluations provide workload-specific evidence (`docs/intel-sources.md:12`). Several proposed external quality and hardware signals remain unwired, so that document is explicitly a scratchpad rather than a committed design (`docs/intel-sources.md:128`).

## Key surfaces

- `somm.llm(project=...)` / `SommLLM.generate(...)` — primary Python call path returning text, provider, and cost telemetry (`docs/index.html:44`).
- `SommLLM.register_workload(...)` — declares privacy, budget, and capability policy (`docs/errors/SOMM_WORKLOAD_UNREGISTERED.md:19`).
- `image_prompt(...)` and `text_prompt(...)` — construct multimodal content-block prompts (`docs/multimodal.md:8`).
- `hooks.register_hook(...)` — registers ordered `pre_call`, `post_call`, or `post_process` callbacks (`docs/plugins.md:118`).
- `ToolCall`, `tools`, `messages`, and `tool_choice` — provider-neutral tool-calling interface (`docs/tool-calling.md:13`).
- `somm serve --project ...` — starts the local dashboard, scheduler, and workers (`docs/index.html:174`).
- `somm compare ... --models ...` — compares providers/models on one prompt (`docs/index.html:181`).
- `somm doctor` — checks processes, cooldowns, migrations, and permission drift (`docs/errors/SOMM_PORT_BUSY.md:11`).
- `somm plugin list` / `somm plugin info` — inspects installed hook plugins (`docs/plugins.md:167`).

## Design decisions

- Telemetry is append-only: late metadata goes into `call_updates`, preserving deterministic audit history (`docs/BLUEPRINT.md:56`).
- Privacy is enforced independently in routing, workers, SQL views, file permissions, and localhost service defaults (`docs/threat-model.md:44`).
- SQLite cooldowns survive restarts, with separate model-level and provider-wide state (`docs/BLUEPRINT.md:187`).
- Budget refusal is fatal and occurs before dispatch, preventing provider fallback from bypassing the cap (`docs/errors/SOMM_BUDGET_EXCEEDED.md:23`).
- Hook events omit prompt and response bodies, and hook failures cannot break the call path (`docs/plugins.md:79`).
- Recommendations expose supporting evidence and require user application rather than automatic rollout (`docs/BLUEPRINT.md:255`).

## One-liner

These docs define and expose somm as a privacy-first local LLM control plane built around immutable SQLite call telemetry.