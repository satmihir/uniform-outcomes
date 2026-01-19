# simulations/common.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import time


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Common experiment parameters shared across all simulations.
    """
    buckets: int
    balls: int
    servers: int = 1  # number of independent schedulers / routers

    def __post_init__(self) -> None:
        if self.buckets <= 0:
            raise ValueError("buckets must be > 0")
        if self.balls < 0:
            raise ValueError("balls must be >= 0")
        if self.servers <= 0:
            raise ValueError("servers must be > 0")


@dataclass(frozen=True)
class SummaryStats:
    """
    Basic summary stats for final bucket counts.
    """
    min: int
    max: int
    mean: float
    std: float  # population stddev


def summarize_counts(counts: List[int]) -> SummaryStats:
    """
    Compute min/max/mean/std over integer counts (population stddev).
    Stddev computed via a two-pass method for clarity.
    """
    if not counts:
        raise ValueError("counts must be non-empty")

    mn = min(counts)
    mx = max(counts)

    n = len(counts)
    total = 0
    for c in counts:
        total += c
    mean = total / n

    # population variance
    var_acc = 0.0
    for c in counts:
        d = c - mean
        var_acc += d * d
    var = var_acc / n
    std = math.sqrt(var)

    return SummaryStats(min=mn, max=mx, mean=mean, std=std)


@dataclass
class ExperimentResult:
    """
    Common return type for all simulations.
    """
    method: str
    spec: ExperimentSpec
    counts: List[int]

    stats: SummaryStats = field(init=False)
    runtime_s: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stats = summarize_counts(self.counts)

        # Sanity: counts should sum to balls
        expected = self.spec.balls
        actual = 0
        for c in self.counts:
            actual += c
        if actual != expected:
            raise ValueError(
                f"counts sum mismatch: expected {expected}, got {actual}"
            )


class Timer:
    """
    Tiny timing helper for simulations.
    Usage:
        with Timer() as t:
            ...
        elapsed = t.elapsed_s
    """
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_s: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.elapsed_s = time.time() - self._start


def common_x_range(results: List[ExperimentResult]) -> Tuple[int, int]:
    """
    Compute a shared (xmin, xmax) across multiple results for 'same x-axis'
    histogram comparisons.
    """
    if not results:
        raise ValueError("results must be non-empty")

    xmin = results[0].stats.min
    xmax = results[0].stats.max
    for r in results[1:]:
        if r.stats.min < xmin:
            xmin = r.stats.min
        if r.stats.max > xmax:
            xmax = r.stats.max
    return xmin, xmax


def format_stats_line(r: ExperimentResult) -> str:
    """
    Human-friendly one-liner for printing in compare tools.
    """
    s = r.stats
    return (
        f"{r.method}: min={s.min}, max={s.max}, mean={s.mean:.3f}, std={s.std:.3f}"
        + (f", runtime={r.runtime_s:.3f}s" if r.runtime_s is not None else "")
    )
