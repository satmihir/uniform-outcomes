# simulations/run.py

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from .common import ExperimentSpec, ExperimentResult
from .methods import get_method


def run_experiment(
    method: str,
    buckets: int,
    balls: int,
    servers: int = 1,
    seed: int = 42,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    """
    Run a single simulation and return an ExperimentResult.

    Parameters
    ----------
    method:
        Name of the method (e.g., 'iid', 'bo2', 'bo2_stale', 'uniform_outcomes').
    buckets:
        Number of buckets.
    balls:
        Number of balls / placements.
    servers:
        Number of independent schedulers/routers (methods may ignore this).
    seed:
        Base RNG seed.
    method_kwargs:
        Optional dict of method-specific kwargs (e.g., {'beta': 1.0}).

    Returns
    -------
    ExperimentResult
    """
    spec = ExperimentSpec(buckets=buckets, balls=balls, servers=servers)
    fn = get_method(method)

    kwargs = method_kwargs or {}
    result = fn(spec, seed, **kwargs)  # type: ignore[arg-type]
    return result


def run_pair(
    method_a: str,
    method_b: str,
    buckets: int,
    balls: int,
    servers: int = 1,
    seed: int = 42,
    method_kwargs_a: Optional[Dict[str, Any]] = None,
    method_kwargs_b: Optional[Dict[str, Any]] = None,
):
    """
    Convenience helper: run two methods under the same spec and seed.

    Returns (result_a, result_b).
    """
    ra = run_experiment(
        method=method_a,
        buckets=buckets,
        balls=balls,
        servers=servers,
        seed=seed,
        method_kwargs=method_kwargs_a,
    )
    rb = run_experiment(
        method=method_b,
        buckets=buckets,
        balls=balls,
        servers=servers,
        seed=seed,
        method_kwargs=method_kwargs_b,
    )
    return ra, rb
