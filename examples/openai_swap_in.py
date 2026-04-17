"""Swap `openai.OpenAI().chat.completions.create()` for somm's shim.

Scenario: your project uses the raw OpenAI SDK. Change the import + one
function call; everything else stays the same.

The shim routes through somm's full provider chain, so you can prefix
the model with a provider to pin, or leave unprefixed to route through
the default chain (ollama first for sovereignty).
"""

from __future__ import annotations

# ---- BEFORE ----------------------------------------------------------
# from openai import OpenAI
# client = OpenAI()
# resp = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "hi"}],
# )
# ---- AFTER -----------------------------------------------------------
from somm.compat import openai_chat_completions as create


def ask(question: str) -> str:
    """Pins to openai/gpt-4o-mini explicitly. Same wire shape as OpenAI's SDK."""
    resp = create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": question},
        ],
        max_tokens=256,
        project="my_project",
        workload="qa",
    )
    # Legacy OpenAI shape works unchanged:
    text = resp.choices[0].message.content
    # somm extras (only present on this compat shim — safe to ignore if legacy
    # code doesn't read them):
    print(f"cost=${resp.somm_cost_usd:.6f}  latency={resp.somm_latency_ms}ms")
    return text


def ask_unrouted(question: str) -> str:
    """No provider prefix → somm's router picks best non-cooled provider."""
    resp = create(
        model="gpt-4o-mini",  # no prefix
        messages=[{"role": "user", "content": question}],
        project="my_project",
        workload="qa",
    )
    return resp.choices[0].message.content


def ask_local_first(question: str) -> str:
    """Pin to ollama for sovereignty / privacy."""
    resp = create(
        model="ollama/gemma4:e4b",
        messages=[{"role": "user", "content": question}],
        project="my_project",
        workload="qa_local",
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    print(ask("What is 2+2?"))
