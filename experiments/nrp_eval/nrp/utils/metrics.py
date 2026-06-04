"""Metrics: per-instance summaries, gap-to-optimum, JSONL writer."""
from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path


def gap_to_optimal(per_instance: Sequence[float], optima: Sequence[float]) -> list[float]:
    """Return gap-to-optimal in percent for each instance.

    gap_pct = (cost - opt) / opt * 100
    """
    if len(per_instance) != len(optima):
        raise ValueError(
            f"Length mismatch: per_instance={len(per_instance)} optima={len(optima)}"
        )
    return [(c - o) / o * 100.0 for c, o in zip(per_instance, optima)]


def tour_length_summary(per_instance: Sequence[float]) -> dict:
    """Return mean/std/min/max/p50/p95/feasible_ratio summary stats.

    Per-instance values are tour lengths (positive).
    """
    finite = [v for v in per_instance if math.isfinite(v)]
    if not finite:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "feasible_ratio": 0.0,
            "n": len(per_instance),
        }
    sorted_vals = sorted(finite)
    n = len(per_instance)
    p50 = sorted_vals[len(sorted_vals) // 2]
    if len(sorted_vals) > 1:
        p95_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.95)))
        p95 = sorted_vals[p95_idx]
    else:
        p95 = sorted_vals[0]
    return {
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "p50": p50,
        "p95": p95,
        "feasible_ratio": len(finite) / n,
        "n": n,
    }


class PerInstanceWriter:
    """Stream per-instance rows to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, default=str) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
