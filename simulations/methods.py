# simulations/methods.py

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

from .common import ExperimentSpec, ExperimentResult, Timer

from src.uniform_outcomes.fast_uniform_outcome_randomizer import FastUniformOutcomeRandomizer


SimFn = Callable[[ExperimentSpec, int], ExperimentResult]


def simulate_iid(spec: ExperimentSpec, seed: int) -> ExperimentResult:
    """
    IID uniform placement: each ball chooses a bucket uniformly at random.
    """
    rng = random.Random(seed)
    counts = [0] * spec.buckets

    with Timer() as t:
        for _ in range(spec.balls):
            b = rng.randrange(spec.buckets)
            counts[b] += 1

    return ExperimentResult(
        method="iid",
        spec=spec,
        counts=counts,
        runtime_s=t.elapsed_s,
        meta={},
    )


def simulate_bo2(spec: ExperimentSpec, seed: int) -> ExperimentResult:
    """
    Power-of-two choices with *fresh global feedback*.

    Each ball samples two buckets uniformly and places into the less-loaded
    one using a single global count array (perfectly fresh, centralized view).

    Note: spec.servers does not change behavior here because the decision is
    based on global truth; multiple schedulers would be equivalent to one.
    """
    rng = random.Random(seed)
    counts = [0] * spec.buckets

    with Timer() as t:
        for _ in range(spec.balls):
            a = rng.randrange(spec.buckets)
            b = rng.randrange(spec.buckets)
            ca = counts[a]
            cb = counts[b]

            if ca < cb:
                counts[a] = ca + 1
            elif cb < ca:
                counts[b] = cb + 1
            else:
                # tie-break randomly
                if rng.random() < 0.5:
                    counts[a] = ca + 1
                else:
                    counts[b] = cb + 1

    return ExperimentResult(
        method="bo2",
        spec=spec,
        counts=counts,
        runtime_s=t.elapsed_s,
        meta={"feedback": "fresh_global"},
    )


def simulate_bo2_stale(spec: ExperimentSpec, seed: int) -> ExperimentResult:
    """
    Power-of-two choices with *stale local views* (multi-scheduler).

    We model 'servers' independent schedulers:
      - Each scheduler has its own local view of counts (locals_[s]).
      - For each ball we pick a scheduler uniformly at random.
      - Scheduler samples two buckets and compares *its local view*.
      - We increment the chosen bucket in:
          - global truth counts
          - that scheduler's local view only

    This captures the "very fast placement from multiple servers" scenario
    where state staleness becomes a problem.
    """
    if spec.servers <= 0:
        raise ValueError("spec.servers must be > 0")

    router_rng = random.Random(seed)
    sched_rngs = [
        random.Random(seed + 1000 * (i + 1))
        for i in range(spec.servers)
    ]

    global_counts = [0] * spec.buckets
    locals_: List[List[int]] = [[0] * spec.buckets for _ in range(spec.servers)]

    with Timer() as t:
        for _ in range(spec.balls):
            s = router_rng.randrange(spec.servers)
            rng = sched_rngs[s]
            local = locals_[s]

            a = rng.randrange(spec.buckets)
            b = rng.randrange(spec.buckets)

            if local[a] < local[b]:
                chosen = a
            elif local[b] < local[a]:
                chosen = b
            else:
                chosen = a if rng.random() < 0.5 else b

            global_counts[chosen] += 1
            local[chosen] += 1

    return ExperimentResult(
        method="bo2_stale",
        spec=spec,
        counts=global_counts,
        runtime_s=t.elapsed_s,
        meta={"feedback": "stale_local", "servers": spec.servers},
    )


def simulate_uniform_outcomes(spec: ExperimentSpec, seed: int, beta: float = 1.0) -> ExperimentResult:
    """
    Uniform-outcomes placement (multi-scheduler) using the FAST variant.

    We model 'servers' independent schedulers:
      - Each scheduler has its own FastUniformOutcomeRandomizer instance.
      - For each ball we pick a scheduler uniformly at random.
      - Scheduler chooses a bucket via uniform-outcomes random placement.
      - Global truth counts record where balls actually land.

    This matches the decentralized, no-shared-state model used in the blog.
    """
    if spec.servers <= 0:
        raise ValueError("spec.servers must be > 0")

    router_rng = random.Random(seed)
    schedulers = [
        FastUniformOutcomeRandomizer(spec.buckets, beta=beta, seed=seed + 1000 * (i + 1))
        for i in range(spec.servers)
    ]

    global_counts = [0] * spec.buckets

    with Timer() as t:
        for _ in range(spec.balls):
            s = router_rng.randrange(spec.servers)
            b = schedulers[s].next()
            global_counts[b] += 1

    return ExperimentResult(
        method="uniform_outcomes",
        spec=spec,
        counts=global_counts,
        runtime_s=t.elapsed_s,
        meta={"beta": beta, "servers": spec.servers},
    )


# --- Registry / dispatch -----------------------------------------------------

def get_method(name: str) -> Callable[..., ExperimentResult]:
    name = name.strip().lower()
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'. Available: {sorted(METHODS.keys())}")
    return METHODS[name]


# METHODS maps method name -> function.
# Note: simulate_uniform_outcomes takes an extra beta parameter; callers can pass it.
METHODS: Dict[str, Callable[..., ExperimentResult]] = {
    "iid": simulate_iid,
    "bo2": simulate_bo2,
    "bo2_stale": simulate_bo2_stale,
    "uniform_outcomes": simulate_uniform_outcomes,
}
