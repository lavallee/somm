"""CI performance budget checks for somm's import and warmed call path."""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from somm.client import SommLLM
from somm.providers.base import ProviderHealth, SommResponse
from somm_core.config import Config
from somm_core.pricing import write_intel

IMPORT_BUDGET_MS = float(os.environ.get("SOMM_IMPORT_BUDGET_MS", "30"))
HOT_PATH_P50_BUDGET_MS = float(os.environ.get("SOMM_HOT_PATH_P50_BUDGET_MS", "1"))


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return SommResponse(
            text="ok",
            model=request.model or "m",
            tokens_in=1,
            tokens_out=1,
            latency_ms=0,
        )

    def stream(self, request):  # pragma: no cover
        yield

    def health(self):
        return ProviderHealth(available=True)

    def models(self):
        return []

    def estimate_tokens(self, text, model):
        return 1


def _percentile(values: list[float], pct: float) -> float:
    idx = max(0, min(len(values) - 1, int((pct * len(values) + 99) // 100) - 1))
    return sorted(values)[idx]


def measure_import_ms(samples: int = 20) -> tuple[float, float]:
    timings: list[float] = []
    for _ in range(samples):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import time; t=time.perf_counter(); import somm; print((time.perf_counter()-t)*1000)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        timings.append(float(proc.stdout.strip()))
    return statistics.median(timings), _percentile(timings, 95)


def measure_hot_path_ms(samples: int = 500) -> tuple[float, float]:
    os.environ.setdefault("SOMM_REGISTRY_ALLOW_TMP", "1")
    with tempfile.TemporaryDirectory(prefix="somm-perf-") as td:
        root = Path(td)
        cfg = Config()
        cfg.project = "perf"
        cfg.mode = "observe"
        cfg.db_dir = root / ".somm"
        cfg.spool_dir = cfg.db_dir / "spool"

        llm = SommLLM(config=cfg, providers=[FakeProvider()])
        try:
            llm.repo.register_workload(name="hot", project=cfg.project)
            write_intel(llm.repo, "fake", "m", 1.0, 1.0, None, None, "perf-budget")
            llm.generate("warm", workload="hot", provider="fake", model="m")

            timings = []
            for idx in range(samples):
                started = time.perf_counter_ns()
                llm.generate(f"prompt {idx}", workload="hot", provider="fake", model="m")
                timings.append((time.perf_counter_ns() - started) / 1_000_000)
        finally:
            llm.close()
            llm.repo.close()
    return statistics.median(timings), _percentile(timings, 95)


def main() -> int:
    import_p50, import_p95 = measure_import_ms()
    hot_p50, hot_p95 = measure_hot_path_ms()

    print(
        "perf budgets: "
        f"import p50={import_p50:.2f}ms p95={import_p95:.2f}ms "
        f"(budget p50<={IMPORT_BUDGET_MS:.2f}ms); "
        f"hot path p50={hot_p50:.3f}ms p95={hot_p95:.3f}ms "
        f"(budget p50<={HOT_PATH_P50_BUDGET_MS:.3f}ms)"
    )

    failed = False
    if import_p50 > IMPORT_BUDGET_MS:
        print(f"import budget exceeded: {import_p50:.2f}ms > {IMPORT_BUDGET_MS:.2f}ms", file=sys.stderr)
        failed = True
    if hot_p50 > HOT_PATH_P50_BUDGET_MS:
        print(
            f"hot-path p50 budget exceeded: {hot_p50:.3f}ms > {HOT_PATH_P50_BUDGET_MS:.3f}ms",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
