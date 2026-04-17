"""Drop-in replacement for a project's existing LLM wrapper class.

Scenario: your project has a `FooLLM` class with a legacy-shape API:

    class FooLLM:
        def generate(self, prompt, system="", max_tokens=256, provider=None):
            ...
            return FooResult(text=..., provider=..., model=...)

To graft somm:

1. Install somm alongside your project:
       uv add --editable ../somm

2. Change ONE import:
       # before:
       from myproject.llm import FooLLM
       # after:
       from somm.compat import GenericLLMCompat as FooLLM

3. Call sites don't need changes. Telemetry + cost + provenance now
   land in `.somm/calls.sqlite` on every call.

This example simulates a pre-existing call site using synthetic names.
"""

from __future__ import annotations

# ---- BEFORE (existing project code, hypothetical) ---------------------
# from myproject.llm import FooLLM
# llm = FooLLM(api_key=os.environ["FOO_API_KEY"])
# ---- AFTER (one-line swap) --------------------------------------------
from somm.compat import GenericLLMCompat as FooLLM


def extract_contacts_from_article(article_text: str) -> dict:
    """Example call site — no somm-specific code needed."""
    llm = FooLLM(project="my_project")
    try:
        result = llm.generate(
            prompt=f"Extract people + emails from: {article_text}",
            system="Return JSON with 'people' and 'emails' arrays.",
            max_tokens=512,
            workload="contact_extract",  # optional but recommended
        )
        # Legacy fields still work:
        print(f"provider={result.provider}  model={result.model}")
        # somm extras available as additional attributes:
        print(f"cost=${result.cost_usd:.6f}  call_id={result.call_id}")
        return {
            "text": result.text,
            "provenance": {
                "call_id": result.call_id,
                "provider": result.provider,
                "model": result.model,
            },
        }
    finally:
        llm.close()


def parallel_extraction(articles: list[str]) -> list[dict]:
    """Uses probe_providers (legacy alias for parallel_slots) to stripe
    work across available providers (e.g., local ollama + openrouter)."""
    llm = FooLLM(project="my_project")
    try:
        slots = llm.probe_providers(len(articles))
        results = []
        for article, provider_slot in zip(articles, slots, strict=True):
            r = llm.generate(
                prompt=f"Extract contacts from: {article}",
                workload="contact_extract",
                provider=provider_slot,
            )
            results.append(
                {
                    "text": r.text,
                    "provider": r.provider,
                    "cost": r.cost_usd,
                }
            )
        return results
    finally:
        llm.close()


if __name__ == "__main__":
    sample = "Dr. Jane Doe (jane@example.org) runs the lab with Alex Smith."
    out = extract_contacts_from_article(sample)
    print(out)
