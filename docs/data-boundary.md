# Data boundary

A data boundary says how far a workload's data may travel. somm classifies
every route it could send a request to (a provider, a model, an endpoint) into
a tier, and refuses any route whose tier is looser than the workload's
boundary. The refusal happens before a request leaves the process.

## The tiers

Strictest first. A workload with boundary `B` may use a route of tier `T` when
`T` is at or above `B` in this table.

| Tier | Meaning | Shipped examples |
|---|---|---|
| `local` | The data never leaves hardware you run | Ollama, or any OpenAI-compatible server, on loopback (`127.0.0.1`, `::1`, `localhost`) or a private address (`10/8`, `172.16/12`, `192.168/16`, `fc00::/7`) |
| `zdr` | A third party processes it under an explicit zero-retention, no-training commitment | Nothing. A ZDR agreement belongs to an account, and can exclude models, so only you can declare it |
| `subscribed` | A US-headquartered, first-party provider you hold an account with, under its standard terms | Anthropic API, OpenAI API (on their own hosts); Claude Code, Codex and ChatGPT seats (*assumed*, see below) |
| `unvetted` | Unknown provenance, non-US jurisdiction, free tiers, or declared training use | Everything else: OpenRouter, Gemini (until you declare a paid key), Perplexity (pending your approval), DeepSeek, MiniMax, Ollama `:cloud` / `-cloud` models, Ollama on a remote host, plugin providers, coding agents that can use any provider |

Rules the classifier follows:

- **Unknown is `unvetted`.** It never guesses upward.
- **A tier belongs to a route, not a provider.** `ollama/qwen3:14b` on your
  machine is `local`; `ollama/gemma4:cloud` through the same daemon is
  `unvetted`, because a cloud-tagged model is run on Ollama's servers. A
  provider pointed at a base URL that is not its own API host (an OpenAI
  client aimed at a gateway) is `unvetted` unless that URL is loopback or
  private.
- **Your file wins.** Tiers are claims about your contracts, so
  `~/.somm/boundary.toml` overrides every shipped default.
- **Every shipped default says why.** Each entry in
  `packages/somm-core/src/somm_core/data/boundary_defaults.toml` has a
  `basis`; an entry that states a vendor fact also has the `source` it was
  read from, a short `quote`, `last_verified` and `confidence`. Only
  `confidence = "high"` entries promote a route above `unvetted`.
- **Assumed tiers.** The seats (`claude-cli`, `codex-cli`, `chatgpt`) ship
  as `subscribed`, but whether a consumer seat trains on your data depends on
  an account setting somm cannot see. `somm boundary routes` and
  `somm boundary audit` print these as `assumed`, not `default`, until you
  declare them in your file.

See every route somm knows, with its tier and reason:

```bash
somm boundary routes          # configured providers + routes seen in the ledger
somm boundary routes --json   # also lists every rule, with source and quote
```

## Setting a boundary

Resolution order for a call, first hit wins:

1. The workload's stored `data_boundary`.
2. `privacy_class=private` gives `local`. A private workload is also capped at
   `local` whatever is stored: it never leaves your hardware.
3. The project default: `SOMM_DATA_BOUNDARY`, then `[projects.<name>]` in
   `boundary.toml`, then `[defaults]` in `boundary.toml`.
4. `unvetted`, the fleet default. It admits every route, which is how somm
   behaved before 0.21.

A per-call `data_boundary=` (and the proxy's `X-Somm-Data-Boundary` header)
may only tighten the result. A looser value is ignored.

```bash
somm workload create triage --data-boundary subscribed   # `add` works too
somm boundary set triage local
somm boundary set triage none                            # back to "derive"
```

```python
llm.register_workload(name="triage", data_boundary="subscribed")
llm.set_data_boundary("triage", "local")
llm.generate(prompt, workload="triage", data_boundary="local")  # tighten one call
```

MCP: `somm_register_workload(name=..., data_boundary="subscribed")`. Workload
read-outs carry `data_boundary`, `effective_data_boundary` and
`data_boundary_source`.

`data_boundary` is not part of a workload's content-addressed id: setting it
never creates a new workload. Passing it to `register_workload` on an
existing workload updates it.

## Writing boundary.toml

The file lives at `~/.somm/boundary.toml`, or wherever `SOMM_BOUNDARY_PATH`
points. somm rereads it when it changes.

```toml
[defaults]
boundary = "subscribed"          # project default for every project on this machine

[projects.fab]
boundary = "local"               # beats [defaults] for one project

# match = "provider" or "provider/model-glob"; the most specific match wins
# (exact model > model glob > provider only).
[[route]]
match = "gemini"
tier = "subscribed"
note = "key belongs to a Cloud project with active billing (paid terms)"

[[route]]
match = "claude-cli"
tier = "subscribed"
note = "checked 2026-11: model-improvement setting is off"

[[route]]
match = "codex-cli"
tier = "unvetted"
note = "'Include environments' is on"

[[route]]
match = "openrouter/anthropic/*"
tier = "zdr"
note = "account-level ZDR-only enforcement on; Covered Models excluded below"

[[route]]
match = "openrouter/anthropic/claude-mythos*"
tier = "subscribed"
note = "Covered Models are retained 30 days even under ZDR"

[[route]]
host = "gpu1.example.ts.net"     # an endpoint host (glob), whatever the provider
tier = "local"
note = "my rented box"
```

Every entry takes a `note`; it is shown as the basis in `somm boundary routes`.
An entry with an unknown tier is treated as `unvetted`. If the file cannot be
parsed, somm ignores all of its promotions and treats every project default as
`local` until it is fixed (`somm doctor` says so): a typo must not loosen
anything.

Workload handlers are matched as `handler:<kind>`. A handler class may also
declare its own tier with a `data_tier` attribute; undeclared is `unvetted`,
so a bounded workload does not consult it (the boundary-checked provider chain
answers instead).

## What happens at a refusal

Every place a request can leave the process calls the same `admit()` check:

| Path | When no admissible route remains |
|---|---|
| `generate` / `agenerate`, routed | Inadmissible providers are dropped before the attempt loop; if none remain, `SommPrivacyViolation` |
| `stream` / `astream` | Same, before the first candidate is opened |
| Pinned call (`provider=`, `model=`) | Raises; a pin does not override a boundary, and `allow_fallback=True` does not rescue it |
| Workload routing policy | The policy's fallback chain is checked the same way |
| OpenRouter roster walk | Checked per roster model; only admissible models are tried |
| Workload handlers | A handler outside the boundary is not called |
| `enable_shadow`, shadow-eval worker | The gold route must be admissible (private workloads stay banned outright); the worker skips a workload whose gold route is not, counting it in `boundary_skipped` |
| Judges | Shadow-eval: an inadmissible judge is refused and noted in the receipt, never called. `somm eval run`: the judge call goes through `generate`, so the item records the refusal as its error |
| Proxy `/v1/messages`, `/v1/chat/completions` | `403` with `SOMM_PRIVACY_VIOLATION`; litellm is never invoked |
| MCP `somm_compare`, CLI `compare` | The model is listed as refused, not silently skipped |
| CLI `bench`, `eval run`, MCP `somm_replay` | Go through `generate`, so the same checks apply |
| Harness attempts (`harnesses.run` or `start` given a `workload` or `data_boundary`, or `run(record=True)`) | Raises before the agent is launched |

A refusal is never retried and never falls through to a looser route. The
error names the boundary, where it came from, and each refused route with its
tier; `describe_error()` gives `data_boundary`, `boundary_source` and
`refused` as fields, with `retryable: false`.

Each refusal is also written to the ledger as a call row, so you can see the
demand a boundary turned away: `outcome = 'unknown'`, `error_kind =
'SommPrivacyViolation'`, `error_detail` starting `refused: data_boundary=`,
an empty provider, zero cost, and `budget_eligible = 0`. (somm uses the
existing `unknown` outcome rather than a new one, because consumers that
switch on the outcome would break on an unfamiliar value.)

Every served call records the tier of its route in `calls.data_tier`.

## The audit workflow

Set boundaries from evidence, before enforcement bites:

```bash
somm boundary audit                    # last 30 days, this project
somm boundary audit --since 7d -v      # list routes per workload
somm boundary audit --project fab --json
```

For each workload the audit shows its declared and effective boundary, the
worst tier that actually carried its data in the window, how many refusals it
had, and `VIOLATION` when the worst tier is looser than the boundary. Every
row that reached a provider counts, whatever its outcome, because a failed
call still sent its prompt. Rows written before 0.21 have no `data_tier`; the
audit classifies them from provider and model at read time (assuming the
configured Ollama URL for Ollama rows) and marks them `classified now`.
`assumed` means the worst tier came from a seat whose real tier depends on an
account setting.

The exit code is `1` when any workload has a violation, so it can gate a
cron job.

A sensible order:

1. Run the audit. Note what each workload has been using.
2. Declare what your contracts actually say in `boundary.toml` (paid Gemini
   key, seat settings, any ZDR agreement).
3. Set a boundary on the workloads that need one, starting with those whose
   worst served tier is already inside it.
4. When the audit is clean, consider a project default with
   `SOMM_DATA_BOUNDARY=subscribed`.

`somm doctor` has a boundary section: the file's state, the project default,
how many workloads have a boundary, and refusals and violations in the last
seven days.

## What is and is not guaranteed

Guaranteed by code:

- somm does not send a request to a route whose tier is outside the
  workload's boundary, on any of the paths above.
- An unclassifiable route is `unvetted`; an unreadable owner file never
  loosens anything.
- Nothing changes on upgrade for a workload that is not private and has no
  boundary, in a project with no default.

Not guaranteed:

- **A tier is a declaration about contracts, not a technical proof.** somm
  controls where it sends a request, not what the far end does with it.
  `subscribed` means "a US first-party provider under its standard terms",
  which may include retention for abuse monitoring. `zdr` is whatever you
  declared it to be.
- Shipped defaults are a snapshot of vendor pages on the `last_verified`
  date. Terms change; re-verify, and keep your own declarations current.
- Code outside somm's call paths (a `pre_call` hook that makes its own
  network calls, a provider plugin that forwards elsewhere, a raw SDK call)
  is not covered. Hooks see the prompt before any boundary check.
- A hostname is never resolved to decide `local`: a box reached by name is
  `unvetted` until you declare it.
- OpenRouter routes are not made `zdr` by somm today; a gateway constraint
  that enforces zero retention on the wire is a planned follow-up.
