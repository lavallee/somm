"""Calibration of decision calls: does a stated probability match how often
the chosen label turns out right?

Pure arithmetic over ``(p, correct)`` pairs, where ``p`` is the probability
the model gave the label it chose and ``correct`` says whether that label was
right. ``somm_core.read.decision_calibration`` feeds it from the ledger.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_BINS = 10


def calibration(pairs: Iterable[tuple[float, bool]], bins: int = DEFAULT_BINS) -> dict:
    """Brier score and expected calibration error for ``(p, correct)`` pairs.

    - ``brier``: mean of ``(p - correct)^2``. 0 is perfect; always answering
      0.5 scores 0.25.
    - ``ece``: split [0, 1] into ``bins`` equal-width bins by ``p``; for each,
      the gap between mean ``p`` and the share correct, weighted by the bin's
      share of pairs (Guo et al. 2017, arXiv 1706.04599).
    - ``accuracy``: share correct. ``bins``: one row per non-empty bin.

    With no pairs, every metric is None and ``n`` is 0.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    rows = [(min(max(float(p), 0.0), 1.0), bool(c)) for p, c in pairs]
    n = len(rows)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "accuracy": None, "bins": []}
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, c in rows:
        groups[min(int(p * bins), bins - 1)].append((p, c))
    table = []
    ece = 0.0
    for i, g in enumerate(groups):
        if not g:
            continue
        mean_p = sum(p for p, _ in g) / len(g)
        acc = sum(1 for _, c in g if c) / len(g)
        ece += len(g) / n * abs(mean_p - acc)
        table.append(
            {"lo": i / bins, "hi": (i + 1) / bins, "n": len(g), "mean_p": mean_p, "accuracy": acc}
        )
    return {
        "n": n,
        "brier": sum((p - (1.0 if c else 0.0)) ** 2 for p, c in rows) / n,
        "ece": ece,
        "accuracy": sum(1 for _, c in rows if c) / n,
        "bins": table,
    }
