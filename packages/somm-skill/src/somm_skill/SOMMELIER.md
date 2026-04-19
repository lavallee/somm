---
name: somm-sommelier
description: Use when the user asks for model advice ("what should I use for X", "good vision models", "cheapest option for long-context") or is choosing between providers/models for an LLM workload. Consults somm's local model intel + prior cross-project decisions, presents ranked options with reasoning, and records the outcome so future sessions can build on it.
---

# somm sommelier — model advisor for coding agents

You are helping the user pick a model for a specific use case. somm has
local model-intel and a cross-project memory of past decisions — use them
instead of guessing from your training data.

This skill assumes `somm-mcp` is configured as an MCP server in this
agent. The relevant tools are `somm_advise`, `somm_record_decision`,
`somm_search_decisions`, plus the existing `somm_recommend`,
`somm_compare`, and `somm_stats`.

## The loop

Three phases: **recall → advise → record**.

### 1. Recall

Before recommending anything, call `somm_search_decisions` with the
user's question. If somm already has a decision for a substantively
similar question — from any project — surface it first. The user may
want to reuse, revisit, or supersede it.

```
somm_search_decisions(
    question="good free vision models on openrouter",
    scope="global",            # decisions mirror globally by default
    limit=5,
)
```

Do not treat past decisions as authoritative — model intel changes. But
*do* acknowledge them: "Last time in `malo` (2026-04-10) we picked
`openrouter/gemma-3-27b-it:free` because … — still relevant?"

### 2. Advise

Call `somm_advise` with the user's constraints extracted from the
conversation:

```
somm_advise(
    question="good free vision models on openrouter",
    capabilities=["vision"],
    providers=["openrouter"],
    free_only=True,
    workload="critique_visual",    # optional — boosts ranking with shadow-eval scores
    limit=8,
)
```

The response carries ranked candidates with `reasons` — a list of
human-readable factors the sommelier weighed. Present them verbatim
rather than restating in your own words; the reasons are
tokenisation-light and calibrated.

**Guidelines for turning the response into a conversation:**

- Show the top 3 with their reasons, not all 8.
- When shadow-eval data exists (`shadow_score` is not null), lead with
  that — it's the only candidate-level quality signal somm has.
- If `prior_decisions` came back, cite them alongside the live
  candidates: "Candidate X matches what we picked in project Y."
- If `candidates` is empty, don't invent — ask the user to loosen
  constraints or run `somm-serve admin refresh-intel`.

### 3. Record

When the user commits to a choice ("let's go with X", "do that"), call
`somm_record_decision` immediately. Do NOT wait for the end of the
session; decisions are the advisory memory other sessions inherit.

```
somm_record_decision(
    question="good free vision models on openrouter",
    rationale="Picked gemma-3-27b because it has the biggest context
               window among free vision models and matches the
               chart-critique workload profile",
    candidates=<candidates list from somm_advise, verbatim>,
    chosen_provider="openrouter",
    chosen_model="google/gemma-3-27b-it:free",
    workload="critique_visual",
    constraints=<constraints dict from somm_advise response>,
    agent="claude-code",
)
```

Rationale is the single most valuable field. Keep it short (≤ 3
sentences) and capture the *reason* (not just the summary — "picked X
because Y beats Z on …"). Future sessions read this.

## When NOT to use sommelier

- **Hard user intent.** If the user says "use Claude Haiku", don't
  pop up a candidate list. Just record a decision with rationale
  "user-specified" if it's a new choice worth remembering.
- **Inside a hot loop.** Sommelier is for choices, not for every call.
  Once a workload has a chosen model, put it in the workload's default
  and move on.
- **Private workloads.** Decisions themselves don't leak prompt bodies,
  but if the *question text* contains sensitive info, phrase the
  question abstractly ("pick a model for sensitive-workload-X") rather
  than verbatim.

## Retrospectives

If the user reports back later ("the gemma-3 pick wasn't great — kept
missing colour contrasts"), search for the original decision and call
`mark_decision_outcome` (surfaced via `somm_record_decision` with
`supersedes=<id>` once that lands, or via the repository directly in
the interim). A retrospective note on a decision is how the memory
self-corrects.

## Example conversation shape

> **User:** I need to add chart screenshots to the critique flow. What
> vision model should I use? Free would be ideal.
>
> **Agent:** *(calls `somm_search_decisions(question="vision model for
> chart critique", scope="global")`)* — no prior decisions found.
>
> *(calls `somm_advise(question="...", capabilities=["vision"], free_only=True)`)*
>
> Here are the three strongest candidates somm knows about right now:
>
> - `openrouter/google/gemma-3-27b-it:free` — free, 128k ctx, vision✓, seen today
> - `openrouter/meta-llama/llama-3.2-11b-vision-instruct:free` — free,
>   128k ctx, vision✓
> - `ollama/llava:13b` — free (local), vision✓
>
> The gemma option has the widest context window, which helps if the
> chart comes with a lot of surrounding text. The ollama option keeps
> the data on your machine — which matters if the workload might end up
> `privacy_class=PRIVATE`. Which trade-off fits?
>
> **User:** Let's try gemma-3 first.
>
> **Agent:** *(calls `somm_record_decision(...)` with rationale "picked
> gemma-3-27b for chart critique — biggest free vision context; keep
> llava as fallback if privacy constraints tighten")*
>
> Recorded. Decision id: `<uuid>`. This will surface next time
> anyone — in this project or another — asks a similar question.
