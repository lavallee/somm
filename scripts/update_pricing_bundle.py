#!/usr/bin/env python3
"""Regenerate somm-core's bundled pricing snapshot from LiteLLM's
model_prices_and_context_window.json.

LiteLLM's price file (MIT, maintained at
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
is the de facto community ground truth for per-token pricing. somm ships a
pruned snapshot of it inside the somm-core wheel so cost tracking works
offline, out of the box, for every provider somm routes to — no network
fetch on the user's hot path, ever.

Run this before a release:

    uv run python scripts/update_pricing_bundle.py            # fetch + write
    uv run python scripts/update_pricing_bundle.py --source /path/to/local.json

The output lands at packages/somm-core/src/somm_core/data/pricing_bundle.json
and is synced into each project DB by somm_core.pricing.sync_bundled_pricing().
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = (
    REPO_ROOT / "packages" / "somm-core" / "src" / "somm_core" / "data" / "pricing_bundle.json"
)

# LiteLLM litellm_provider values → somm provider names. Only providers somm
# actually routes to are bundled; everything else is pruned.
PROVIDER_MAP = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "openrouter": "openrouter",
    "minimax": "minimax",
}

# Modes worth bundling. Embedding pricing matters for the openai path;
# everything else somm calls is chat-shaped.
MODES = {"chat", "responses", "embedding"}

# Hand-curated rows LiteLLM doesn't carry (native-API model ids somm's
# adapters use). Same tuple shape as the generated entries.
EXTRA_ROWS: list[dict] = [
    # Minimax native API ids (LiteLLM tracks the openrouter-side ids).
    {"provider": "minimax", "model": "MiniMax-M2.7", "price_in_per_1m": 0.3,
     "price_out_per_1m": 1.2, "context_window": 1000192,
     "capabilities": {"tools": True}},
]


def _strip_provider_prefix(key: str, litellm_provider: str) -> str:
    prefix = f"{litellm_provider}/"
    return key[len(prefix):] if key.startswith(prefix) else key


def build_bundle(raw: dict) -> dict:
    models: list[dict] = []
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        somm_provider = PROVIDER_MAP.get(spec.get("litellm_provider", ""))
        if somm_provider is None:
            continue
        if spec.get("mode") not in MODES:
            continue
        model = _strip_provider_prefix(key, spec["litellm_provider"])
        price_in = float(spec.get("input_cost_per_token") or 0.0) * 1_000_000
        price_out = float(spec.get("output_cost_per_token") or 0.0) * 1_000_000
        caps: dict = {}
        if spec.get("supports_function_calling") is True:
            caps["tools"] = True
        if spec.get("supports_vision") is True:
            caps["vision"] = True
        entry = {
            "provider": somm_provider,
            "model": model,
            "price_in_per_1m": round(price_in, 6),
            "price_out_per_1m": round(price_out, 6),
            "context_window": spec.get("max_input_tokens"),
        }
        if caps:
            entry["capabilities"] = caps
        models.append(entry)

    models.extend(EXTRA_ROWS)
    models.sort(key=lambda m: (m["provider"], m["model"]))
    return {
        "generated_on": date.today().isoformat(),
        "source": LITELLM_URL,
        "license": "MIT (LiteLLM); see source",
        "models": models,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="local LiteLLM json instead of fetching")
    args = ap.parse_args()

    if args.source:
        raw = json.loads(Path(args.source).read_text())
    else:
        print(f"fetching {LITELLM_URL} …", file=sys.stderr)
        with urllib.request.urlopen(LITELLM_URL, timeout=60) as r:
            raw = json.loads(r.read().decode())

    bundle = build_bundle(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(bundle, indent=1) + "\n")
    by_provider: dict[str, int] = {}
    for m in bundle["models"]:
        by_provider[m["provider"]] = by_provider.get(m["provider"], 0) + 1
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}: {len(bundle['models'])} models")
    for p, n in sorted(by_provider.items()):
        print(f"  {p:<12} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
