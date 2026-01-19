# simulations/compare.py

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from .common import common_x_range, format_stats_line
from .run import run_experiment


# Keep the tool intentionally opinionated:
# - beta is fixed (so the CLI stays minimal)
# - seed is fixed unless you edit the file
DEFAULT_SEED = 42
DEFAULT_BETA = 1.0


def _method_kwargs(method_name: str):
    """
    Only the 'uniform_outcomes' method needs extra kwargs (beta).
    Keep this internal so the CLI stays tiny.
    """
    name = method_name.strip().lower()
    if name == "uniform_outcomes":
        return {"beta": DEFAULT_BETA}
    return {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two placement methods via Monte Carlo (same x-axis plots)."
    )
    parser.add_argument("--method-a", required=True, help="e.g. iid | bo2 | bo2_stale | uniform_outcomes")
    parser.add_argument("--method-b", required=True, help="e.g. iid | bo2 | bo2_stale | uniform_outcomes")
    parser.add_argument("--buckets", type=int, required=True, help="number of buckets")
    parser.add_argument("--balls", type=int, required=True, help="number of balls")
    parser.add_argument("--servers", type=int, required=True, help="number of schedulers/routers")

    args = parser.parse_args(argv)

    # Run both experiments
    ra = run_experiment(
        method=args.method_a,
        buckets=args.buckets,
        balls=args.balls,
        servers=args.servers,
        seed=DEFAULT_SEED,
        method_kwargs=_method_kwargs(args.method_a),
    )
    rb = run_experiment(
        method=args.method_b,
        buckets=args.buckets,
        balls=args.balls,
        servers=args.servers,
        seed=DEFAULT_SEED,
        method_kwargs=_method_kwargs(args.method_b),
    )

    # Print stats
    print(format_stats_line(ra))
    print(format_stats_line(rb))

    # Plot with same x-axis
    xmin, xmax = common_x_range([ra, rb])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(ra.counts, bins=60, range=(xmin, xmax))
    plt.title(ra.method)
    plt.xlabel("Balls per bucket")
    plt.ylabel("Number of buckets")
    plt.xlim(xmin, xmax)

    plt.subplot(1, 2, 2)
    plt.hist(rb.counts, bins=60, range=(xmin, xmax))
    plt.title(rb.method)
    plt.xlabel("Balls per bucket")
    plt.xlim(xmin, xmax)

    plt.suptitle(
        f"Compare: {ra.method} vs {rb.method}  "
        f"(buckets={args.buckets}, balls={args.balls}, servers={args.servers})"
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
