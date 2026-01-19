import random
import math
from typing import List, Optional


class UniformOutcomeRandomizer:
    """
    UniformOutcomeRandomizer (Reference Implementation)

    This class generates outcomes from a fairness-biased random distribution.
    On each call to next(), outcome i is chosen with probability proportional to:

        exp(-beta * (count_i - min(counts)))

    In other words, bias is applied ONLY to excess above the global baseline.
    This suppresses imbalance while preserving randomness.

    IMPORTANT NOTES:

    - This implementation is intentionally O(k) per call, where k is the number
      of buckets.
    - It recomputes the full probability distribution on every placement.
    - This is done for clarity, not performance.

    The purpose of this class is to act as an *executable specification* of the
    algorithm. Real systems should reimplement the logic with appropriate data
    structures and synchronization.

    This code is:
      - single-threaded
      - not thread-safe
      - not production-ready

    If you are using this directly in production, you are almost certainly
    doing the wrong thing.
    """

    def __init__(
        self,
        num_buckets: int,
        beta: float = 1.0,
        seed: Optional[int] = None,
    ):
        if num_buckets <= 0:
            raise ValueError("num_buckets must be > 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")

        self.beta = beta
        self.counts: List[int] = [0] * num_buckets

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------

    def next(self) -> int:
        """
        Sample the next bucket index according to the fairness-biased
        distribution and update internal state.
        """
        # Baseline removal (Tetris row clearing)
        baseline = min(self.counts)

        # Compute unnormalized weights
        weights = []
        total_weight = 0.0

        for c in self.counts:
            excess = c - baseline
            w = math.exp(-self.beta * excess)
            weights.append(w)
            total_weight += w

        # Sample from the categorical distribution
        r = random.random() * total_weight
        cumulative = 0.0

        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                self.counts[i] += 1
                return i

        # Numerical fallback (should never happen)
        last = len(self.counts) - 1
        self.counts[last] += 1
        return last

    # ------------------------------------------------------------
    # Autoscaling helpers (structural changes only)
    # ------------------------------------------------------------

    def add_buckets(self, n: int) -> None:
        """
        Add n new buckets (autoscaling up).

        New buckets start at count = 0 and naturally integrate into the
        distribution via baseline removal.
        """
        if n <= 0:
            raise ValueError("n must be > 0")

        self.counts.extend([0] * n)

    def remove_bucket(self, index: int) -> None:
        """
        Remove a bucket by index (autoscaling down).

        This deletes the bucket and its history. Remaining indices shift.
        """
        if index < 0 or index >= len(self.counts):
            raise IndexError("bucket index out of range")

        del self.counts[index]

        if not self.counts:
            raise RuntimeError("cannot remove the last remaining bucket")

    # ------------------------------------------------------------
    # Introspection (read-only)
    # ------------------------------------------------------------

    def num_buckets(self) -> int:
        return len(self.counts)

    def snapshot_counts(self) -> List[int]:
        """
        Return a copy of the internal counts for inspection/debugging.
        """
        return list(self.counts)
