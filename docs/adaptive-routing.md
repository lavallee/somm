# Adaptive routing and ChatGPT subscription access

Adaptive routing is opt-in per workload. Use `model="somm/auto"` with a validated
`policy.adaptive` object; existing explicit provider/model pins retain their meaning.
The library (`generate`, `agenerate`, buffered `stream`) and the OpenAI-compatible
`POST /v1/chat/completions` proxy share the same decision function.

```json
{
  "adaptive": {
    "fast": {"provider": "chatgpt", "model": "gpt-6-luna", "reasoning_effort": "none"},
    "strong": {"provider": "chatgpt", "model": "gpt-6.1-sol", "reasoning_effort": "high"},
    "fallback": {"provider": "minimax", "model": "MiniMax-M3"},
    "fast_patterns": ["add .+ to (?:the )?grocer(?:y list|ies)"],
    "weekly_limit_percent": 90,
    "quota_max_age_seconds": 300
  }
}
```

Set the policy with `SommLLM.set_workload_policy(workload, policy)` or the existing
workload policy API/CLI. Replace the example models with models available to the
signed-in account, and confirm real tool inference before enabling a production route.
A bundled Codex model list is not proof of account entitlement.

## Decisions

This first router is deliberately deterministic. Greetings and configured full-match
patterns take the fast tier. Biographies, career attribution, research, verification,
and medical/legal/financial questions take the strong tier. Everything ambiguous takes
the strong tier too. The fast patterns are application configuration, not a claim
that Somm understands every task. Do not use a broad catch-all pattern.

Only human text classifies the task. Tool results do not become new user requests;
short confirmations inherit the preceding substantive human request. A new explicit
simple request can leave an earlier research task. No prompt or conversation text is
saved in routing telemetry.

For a ChatGPT target, usage comes from Codex's supported
`account/rateLimits/read` RPC. Identify the weekly window by `windowDurationMins=10080`,
not by the field name `secondary`. The probe caches for 60 seconds. Missing, expired,
invalid, or stale weekly metadata conserves quota rather than assuming zero usage.
Five-hour usage, credit balances, and API dollar prices do not participate in this
weekly policy. Below 90% the selected ChatGPT model runs. At or above 90%, fast work
uses the configured non-ChatGPT fallback and strong work is deferred. The reset time
comes from the provider, not a locally invented week boundary. Concurrent clients can
consume allowance between samples; this is a conservation threshold, not a reservation.

A deferral raises `SommDeferred` in the library. The proxy returns HTTP 429 with
`error.code=SOMM_WORK_DEFERRED` and `error.routing` (including reset metadata). **No
inference was attempted and Somm did not enqueue the task.** The application must
retain the request, tell the person it is deferred, and decide when/how to resume.
Never treat that response as a completed answer or silently send strong work to the
fallback model. Provider/authentication failures also do not authorize a downgrade.

`POST /api/routing/decide` accepts `messages` plus the same workload/project headers
and service authentication. It returns the decision (including deferral) without
inference or prompt storage, for application preflight and user feedback. The actual
inference request checks again, so preflight is not a reservation.

Each served adaptive call records the selected tier, reason, target, quota observation,
and actual served provider/model. `somm tail` / call search include `routing_json`.
ChatGPT costs are marked `included` with source `chatgpt-subscription` and zero marginal
API charge; this does not estimate subscription economics. API dollar budget gates
remain active for MiniMax and other API targets.

## Connect ChatGPT

Install the optional dependencies: `pip install 'somm[chatgpt]'` (or `uv sync --extra
chatgpt` in the workspace). Run `somm chatgpt login` on a machine with a browser. It
uses OpenAI's supported Sign in with ChatGPT registration, loopback callback, PKCE,
state, nonce, verified OIDC identity, and explicit plan-usage permission. The callback
port defaults to 1456 and can be forwarded for a headless installation. Authorization
must complete on the host reachable at the browser's callback URL.

Somm owns its own credential record: `~/.config/somm/chatgpt.json`, overridden by
`SOMM_CHATGPT_AUTH_FILE`. It never opens another application's credential file or
uses private ChatGPT HTTP endpoints. Files are atomically written with mode 0600;
rotating refreshes serialize across processes with a file lock. Keep credentials and
callback URLs out of logs, command histories, source control, and research artifacts.
For a remote VM, transfer the protected Somm record over SSH after local authorization,
retain the VM's own host identifier, and let only the destination own future refreshes.

`somm chatgpt status` exposes only connection/expiry status. `somm chatgpt models`
queries the public account model catalog. `somm chatgpt quota` uses the separate
first-party Codex account session; if stale, run `codex login --device-auth` on that
host. Ensure both connections belong to the intended account/workspace. A quota login
alone does not activate Somm inference.

The `chatgpt` provider is pinned-only and never joins the default API fallback chain.
It supports function tools, multimodal input, and encrypted reasoning replay through
the public Responses API, while preserving the proxy's Chat Completions wire shape.
HTTP 200 is insufficient: only `response.completed` completes a call. Interrupted,
failed, incomplete, unauthorized, and rate-limited streams remain failures. Proxy
streaming is not supported; library adaptive streaming buffers one completed answer.
The SIWC preview requires developer instructions and additional-tools input items;
the adapter translates system messages and flat function declarations accordingly.
The route rejects `max_output_tokens`, so Chat Completions `max_tokens` is an
unsupported output hint here and is omitted; the transport timeout still applies.
Replay preserves original message phases, function calls, and encrypted reasoning.
See [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).

## Quality and alternatives

A stronger model cannot repair falsely attributed memory by itself. Biographical
work needs evidence explicitly about the subject, peer/source attribution, and
owner corrections applied to memory. Evaluate with fictional subjects and employers,
including a peer's employment appearing near the subject's feed, a negative owner
correction, a short continuation, and multi-turn tool use. Never commit household
records as fixtures.

RouteLLM and FrugalGPT support the broader case for routing/cascades, not a guarantee
for current models or this household. A learned classifier needs task-specific evals.
LiteLLM auto-routing is a broader beta integration; Hermes's JEV plugin adds a separate
classifier account and context sharing. OpenRouter Auto adds another hosted processor
and API billing. A conservative workload policy is the smaller first implementation
for an existing Somm gateway and a ChatGPT subscription.

Primary sources reviewed 2026-11-09:

- [OpenAI Sign in with ChatGPT](https://developers.openai.com/siwc/token-sharing-open-source),
  [sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in),
  [models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference),
  [remote hosts](https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms).
- [Codex app-server account RPC](https://learn.chatgpt.com/docs/app-server),
  [subscription pricing and limits](https://developers.openai.com/codex/pricing).
- [Model selection](https://developers.openai.com/api/docs/guides/model-selection).
- [LiteLLM automatic routing](https://docs.litellm.ai/docs/proxy/auto_routing),
  [Hermes JEV routing plugin](https://hermes-agent.nousresearch.com/marketplace/hermes-jev-routing),
  [OpenRouter Auto](https://openrouter.ai/docs/guides/routing/routers/auto-router).
- [RouteLLM](https://arxiv.org/abs/2406.18665),
  [FrugalGPT](https://arxiv.org/abs/2305.05176).

## Failure and recovery contract

The Chat Completions proxy preserves typed provider failures instead of converting
all of them into HTTP 502. Invalid input returns 400, authentication failures 401
(or a preserved upstream denial such as 403), rate limits 429, transport timeouts
504, and transient upstream failures 502. `error.code` is Somm's canonical code;
`error.retryable` and optional `error.retry_after_s` are the runner contract.
`x-should-retry` mirrors retryability for compatible SDKs. A 429 deferral or API
budget refusal explicitly says `retryable: false`; waiting and sending the identical
request repeatedly will not repair it.

A failed inference response identifies its `call_id`, provider, selected model,
and adaptive route when present. The same ID appears in `x-somm-call-id` and the
ledger. Safe machine fields (`provider_code`, `provider_type`, `provider_param`,
`provider_request_id`) survive translation when the adapter supplies them. Clients
should use these fields rather than parsing provider prose.

This HTTP boundary neither retries inference nor chooses another model. Resume a
transient failure with the original conversation, including completed tool results;
do not repeat household mutations to reconstruct context. An unrecoverable input or
authorization error needs correction before another attempt. Provider-side replay
normalization may repair a known transcript mismatch without another inference call;
its `somm_recovery` marker is retained in `routing_json.recovery` for explicit pins
as well as adaptive calls, with counts rather than prompt/tool-result contents.


## Normalized tool replay recovery (0.20.1)

Some harnesses expand one model batch into multiple tool calls and completed results.
Somm compares normalized call IDs, names and semantic arguments with its opaque replay
items before inference. If they differ, it rebuilds the same-model request from the
visible conversation, preserving every completed result and message phase while
omitting stale opaque reasoning for that request. It neither executes a tool nor
creates a replacement result. Unmatched, duplicate pending, or missing results stop
with a fatal bad-request error rather than a provider-outage retry loop.

The successful response's `somm_recovery` marker and call `routing_json.recovery`
record the repair kind and preserved-result count, without prompt content. The model
and weekly conservation policy remain unchanged. A response with `code:null` still
uses its structured error type; malformed input is distinct from transient provider
unavailability. This recovery resumes the existing tool-result context; application
workflow custody remains with the calling harness.
