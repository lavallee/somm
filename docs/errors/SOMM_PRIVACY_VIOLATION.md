# SOMM_PRIVACY_VIOLATION

**Problem.** A request was refused because no route within its workload's
data boundary remained. Nothing was sent to any provider.

The message names the boundary, where it came from, and every route refused
with its tier:

```
SOMM_PRIVACY_VIOLATION

Problem: workload 'medical_extract' has data boundary 'local' (from
privacy_class=private), and no admissible route remains for the routing chain.
Cause: refused routes: openai/gpt-x (subscribed), deepseek/deepseek-chat (unvetted). ...
```

`describe_error(exc)` returns the same as fields: `code`, `data_boundary`,
`boundary_source`, `refused` (each with `provider`, `model`, `tier`, `basis`,
`origin`) and `retryable: false`. The proxy returns it as HTTP 403.

**Common triggers.**

- A `privacy_class=PRIVATE` workload (boundary `local`) called when no local
  route is configured or the only candidates are cloud providers.
- A pinned call (`provider=`/`model=`) to a route outside the boundary. A pin
  does not override a boundary, and `allow_fallback=True` does not rescue it.
- `SommLLM.enable_shadow()` on a PRIVATE workload (always refused), or with a
  gold route outside the workload's boundary.
- A proxy request whose target model is outside the workload's boundary.
- A harness attempt (`somm.harnesses.run/start`) for a bounded workload whose
  coding agent runs at a looser tier.
- MCP `somm_replay` of a PRIVATE workload's call.

**Why.** The boundary is the furthest a workload's data may go (see
[docs/data-boundary.md](../data-boundary.md)). A refusal is never retried
and never falls through to a looser route. It is recorded in the ledger as a
call row (`outcome='unknown'`, `error_kind='SommPrivacyViolation'`,
`error_detail` starting `refused: data_boundary=`), so `somm boundary audit`
can show the demand a boundary turned away.

**Fix.** Find out which route you expected to use:

```bash
somm boundary routes        # every route with its tier and why
somm workload show <name>   # stored and effective boundary
```

Then, depending on intent:

### A. Add a route within the boundary

For `local`, run a local model (Ollama on loopback) and make sure it is in
the chain (`SOMM_PROVIDER_ORDER`). Ollama `:cloud` / `-cloud` models are not
local even through a local daemon.

### B. The route is actually within your contract

If your account terms put the route in a stricter tier than somm assumes (a
paid Gemini key, a seat with training off, a ZDR agreement), declare it in
`~/.somm/boundary.toml`:

```toml
[[route]]
match = "gemini"
tier = "subscribed"
note = "billing-enabled Cloud project"
```

### C. The workload's boundary is too tight

```bash
somm boundary set <workload> subscribed
```

A `privacy_class=PRIVATE` workload is always capped at `local`; change its
privacy class if it does not handle sensitive content.

### D. Leave it

The refused request is exactly the one the boundary exists to prevent.

**Related.**
- [docs/data-boundary.md](../data-boundary.md): tiers, boundary.toml, audit.
- [`SOMM_WORKLOAD_UNREGISTERED`](./SOMM_WORKLOAD_UNREGISTERED.md).
