# SOMM_NO_CAPABLE_PROVIDER

**Problem.** The request requires a capability (e.g. `vision`) that no
provider in the routing chain can serve with its current default model.
Raised *before* any network call so you can fix the gap explicitly
rather than discover it via a late 400.

**Why.** Common triggers:

- A vision prompt hit a project configured with only text-capable
  providers (e.g. `ollama/gemma4:e4b`).
- `SOMM_PROVIDER_ORDER` limited the chain to providers without a
  vision-capable default model.
- The workload declared `capabilities_required=["long_context"]` but
  the default model for each provider has a smaller context window.

**Fix.**

1. Inspect what was skipped. The exception carries a `skipped` list of
   `(provider, model, reason)` triples:
   ```python
   from somm.errors import SommNoCapableProvider
   try:
       llm.generate(prompt=image_prompt(...), workload="vision")
   except SommNoCapableProvider as e:
       for provider, model, reason in e.skipped:
           print(provider, model, reason)
       # → openai   gpt-4o-mini   missing_capability:vision   (hypothetical)
   ```

2. Point an existing provider at a capable model:
   ```bash
   export SOMM_ANTHROPIC_MODEL=claude-opus-4-7    # vision-capable
   export SOMM_OPENAI_MODEL=gpt-4o                # vision-capable
   ```

3. Or add a capable provider to the chain:
   ```bash
   export OPENROUTER_API_KEY=...                  # many vision-capable roster models
   export ANTHROPIC_API_KEY=...
   ```

4. If you're certain a model *is* capable but somm doesn't know yet,
   refresh model intel:
   ```bash
   somm-serve admin refresh-intel
   ```
   Or write it in manually:
   ```python
   from somm_core.pricing import write_intel
   write_intel(repo, provider="custom", model="my-vlm",
               price_in_per_1m=0, price_out_per_1m=0,
               context_window=None,
               capabilities={"vision": True}, source="manual")
   ```

**Note.** Capability values of `None` ("unknown") in
`model_intel.capabilities_json` **do not** block — the router allows
the provider to try. Only explicit `False` filters a model out. This
preserves the "let the provider try when unsure" default.

**Related.**

- [`SOMM_PROVIDERS_EXHAUSTED`](./SOMM_PROVIDERS_EXHAUSTED.md) — every
  chain member is cooling, not a capability mismatch.
