# SOMM_WORKLOAD_UNREGISTERED

**Problem.** A call was made with a workload name that isn't registered
in the project, and the library is in strict mode.

**Why this check exists.** Telemetry is meaningful only when workloads
have first-class identity (schema, privacy class, quality criteria).
Strict mode refuses ad-hoc workloads to keep downstream analysis (agent
recommendations, shadow-eval, cost allocation) clean.

**Fix.**

Register the workload once at app startup:

```python
import somm
from somm_core.models import PrivacyClass

llm = somm.llm(project="my_project")
llm.register_workload(
    name="contact_extract",
    description="Extract person names + emails from unstructured text",
    privacy_class=PrivacyClass.INTERNAL,
    budget_cap_usd_daily=0.50,  # optional
)
```

Or via the MCP tool:

```python
# from a coding agent session:
somm_register_workload(name="contact_extract", privacy_class="internal")
```

If you're prototyping and don't want registration ceremony yet, switch to
observe mode (the default):

```bash
export SOMM_MODE=observe   # in your project env
```

In observe mode, unknown workloads get auto-registered on first use.

**Related.**
- [`SOMM_PRIVACY_VIOLATION`](./SOMM_PRIVACY_VIOLATION.md) — privacy gate
  that triggers after successful registration.
