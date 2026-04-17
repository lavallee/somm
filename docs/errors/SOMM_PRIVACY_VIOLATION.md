# SOMM_PRIVACY_VIOLATION

**Problem.** Code attempted an operation on a `privacy_class=PRIVATE`
workload that would egress its prompt or response to an upstream
provider (a commercial API, or any provider outside the local node).

**Common triggers.**

- `somm_replay(call_id, with_provider=X, with_model=Y)` against a call
  whose workload is PRIVATE.
- `SommLLM.enable_shadow(workload, gold_provider=..., gold_model=...)`
  on a PRIVATE workload — the shadow-eval worker would re-send the
  prompt to a non-local gold model.
- (Defense in depth) The router refuses to route a PRIVATE workload
  to any provider other than local ollama.

**Why.** `privacy_class=PRIVATE` is somm's strongest signal that a
workload handles sensitive content (PHI, PII, private communications,
credentials, IP). The defense is enforced at multiple layers:

1. The `shadow_candidates` view excludes private workloads.
2. The `ShadowEvalWorker` double-checks at worker time.
3. `SommLLM.enable_shadow()` raises before writing the config.
4. The router refuses upstream egress at call time.

**Fix.**

Three options, depending on intent:

### A. The workload should not be PRIVATE.

If the classification was conservative and the prompts are in fact
internal-only (not personally sensitive), downgrade:

```python
from somm_core.models import PrivacyClass
llm.register_workload(
    name="contact_extract",
    privacy_class=PrivacyClass.INTERNAL,  # was PRIVATE
)
```

(`register_workload` is idempotent on `(name + schemas)` but updates
fields on re-call via the underlying `repo.register_workload`. You
may need to manually `UPDATE workloads SET privacy_class = 'internal'`
if the conflict path doesn't cover your case — see the
`set_shadow_config` pattern.)

### B. You want the operation but only on local models.

Replay / compare using `provider="ollama"` explicitly:

```python
# MCP:
somm_replay(call_id="...", with_provider="ollama", with_model="gemma4:e4b")

# Python:
llm.generate(prompt, workload="secret_workload", provider="ollama")
```

The router permits ollama-only paths on PRIVATE workloads.

### C. Leave as-is and respect the gate.

The refused operation is exactly the one `privacy_class=PRIVATE` exists
to prevent. If the prompt really is sensitive, the right answer is not
to run it. Consider: do you need shadow-eval on this workload? If not,
just accept that its quality signal will be based on structural checks
+ your own spot reviews.

**Related.**
- [`SOMM_WORKLOAD_UNREGISTERED`](./SOMM_WORKLOAD_UNREGISTERED.md) —
  earlier in the lifecycle; happens before privacy_class is set.
