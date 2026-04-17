"""Register a privacy_class=PRIVATE workload — bans upstream egress.

Use this pattern for sensitive prompts (PII, health data, private
communications). somm's router refuses to call any non-local provider
for calls tagged with this workload, AND shadow-eval workers skip it
entirely.
"""

from __future__ import annotations

import somm
from somm_core.models import PrivacyClass


def main():
    llm = somm.llm(project="my_project")
    try:
        # Register the workload as PRIVATE once (outside the hot path).
        llm.repo.register_workload(
            name="medical_extract",
            project="my_project",
            description="Extract medical entities — PHI handling.",
            privacy_class=PrivacyClass.PRIVATE,
            budget_cap_usd_daily=0.0,  # belt-and-suspenders
        )

        # Now every call on this workload is route-restricted to local providers.
        # If ollama is down and openrouter/anthropic/openai are the only
        # remaining options, the call will fail LOUD rather than egress.
        result = llm.generate(
            prompt="Patient presented with fever and cough, ...",
            workload="medical_extract",
        )
        print(result.text, result.provider)  # provider will be "ollama" or fail

        # enable_shadow() on this workload raises SommPrivacyViolation:
        try:
            llm.enable_shadow(
                workload="medical_extract",
                gold_provider="anthropic",
                gold_model="claude-opus-4-7",
            )
        except somm.SommPrivacyViolation as e:  # type: ignore[attr-defined]
            print("correctly refused:", e.code)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
