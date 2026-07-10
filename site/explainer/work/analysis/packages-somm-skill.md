## Purpose

`somm-skill` packages onboarding instructions for coding agents working on Python projects that use Somm. Its two Markdown resources teach agents how to make observable LLM calls and how to select models using local telemetry and prior decisions rather than provider-specific integrations or guesses (`packages/somm-skill/README.md:3`, `packages/somm-skill/README.md:6`).

## How it works

`SKILL.md` directs agents to create an `somm.llm()` client, tag every generation with a stable workload, register that workload with privacy and budget controls, and mark results with typed outcomes (`packages/somm-skill/src/somm_skill/SKILL.md:24`, `packages/somm-skill/src/somm_skill/SKILL.md:44`, `packages/somm-skill/src/somm_skill/SKILL.md:103`). Quality-sensitive workloads can enable sampled shadow evaluation, while in-process workers or `somm serve` run grading, model-intelligence refreshes, and recommendation generation (`packages/somm-skill/src/somm_skill/SKILL.md:66`, `packages/somm-skill/src/somm_skill/SKILL.md:160`).

`SOMMELIER.md` defines a separate model-selection lifecycle: recall related cross-project decisions, request ranked live candidates through `somm_advise`, then record the user’s committed choice and rationale (`packages/somm-skill/src/somm_skill/SOMMELIER.md:17`, `packages/somm-skill/src/somm_skill/SOMMELIER.md:21`, `packages/somm-skill/src/somm_skill/SOMMELIER.md:40`, `packages/somm-skill/src/somm_skill/SOMMELIER.md:110`). The Python package contains no runtime implementation beyond its module docstring; consumers load or copy the bundled Markdown resources, including through `importlib.resources` in the documented Claude Code installation flow (`packages/somm-skill/README.md:19`, `packages/somm-skill/src/somm_skill/__init__.py:1`).

## Key surfaces

- `SKILL.md` — canonical agent guidance for implementing and operating Somm-backed LLM calls (`packages/somm-skill/src/somm_skill/SKILL.md:6`).
- `SOMMELIER.md` — MCP-based model-advisory workflow built around recall, ranking, and decision recording (`packages/somm-skill/src/somm_skill/SOMMELIER.md:6`).
- `somm.llm()` / `llm.generate()` — recommended application entry point for telemetry-tagged generation (`packages/somm-skill/src/somm_skill/SKILL.md:24`).
- `somm_search_decisions` → `somm_advise` → `somm_record_decision` — the model-selection tool sequence (`packages/somm-skill/src/somm_skill/SOMMELIER.md:12`).
- `somm` CLI — debugging and operations commands for status, generation, plans, evaluation, spend, health, and telemetry recovery (`packages/somm-skill/src/somm_skill/SKILL.md:185`).

## Design decisions

- The package is dependency-free and ships guidance as package resources, keeping it portable across Claude, Codex, Cursor, Windsurf, and other agent packaging formats (`packages/somm-skill/pyproject.toml:4`, `packages/somm-skill/pyproject.toml:16`, `packages/somm-skill/README.md:34`).
- Workloads—not providers—are the stable telemetry identity, separating task semantics from routing choices (`packages/somm-skill/src/somm_skill/SKILL.md:44`).
- Prompt-body capture requires explicit shadow-evaluation configuration and is always disabled for private workloads (`packages/somm-skill/src/somm_skill/SKILL.md:83`).
- CLI subscription seats are pinned-only, preserving them for low-volume or grading work instead of exposing scarce quota to hot-loop routing (`packages/somm-skill/src/somm_skill/SKILL.md:124`).
- Past model decisions inform recommendations but are deliberately non-authoritative because model intelligence changes (`packages/somm-skill/src/somm_skill/SOMMELIER.md:36`).

## One-liner

`somm-skill` is the portable agent playbook that turns Somm usage into a consistent telemetry, evaluation, routing, and model-decision loop.