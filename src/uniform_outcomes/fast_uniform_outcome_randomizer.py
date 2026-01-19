import random
import math
from typing import Dict, List, Optional


class FastUniformOutcomeRandomizer:
    """
    FastUniformOutcomeRandomizer (Engineering Sketch)

    This class samples the SAME distribution as the reference implementation:

        P(i) ∝ exp(-beta * (count_i - min(counts)))

    but avoids O(k) work per placement by exploiting the fact that
    excess values remain small when the algorithm is working.

    Key ideas:
      - Buckets are grouped by absolute count ("height levels")
      - We sample a level first, then a bucket within that level
      - The baseline (minimum count) is tracked lazily

    IMPORTANT:
      - This is not production-ready
      - Single-threaded
      - Not thread-safe
      - Intended as a demonstration of feasibility, not a drop-in library
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

        self.beta = float(beta)
        self._rng = random.Random(seed)

        # bucket_count[b] = absolute count for bucket b
        self.bucket_count: List[int] = [0] * num_buckets

        # buckets_by_count[c] = list of bucket IDs with absolute count == c
        self.buckets_by_count: Dict[int, List[int]] = {
            0: list(range(num_buckets))
        }

        # bucket_pos[b] = index of b inside buckets_by_count[bucket_count[b]]
        self.bucket_pos: List[int] = list(range(num_buckets))

        # Current minimum absolute count present
        self.min_count: int = 0

        # Cache for exp(-beta * delta) where delta is small
        self._exp_cache: Dict[int, float] = {0: 1.0}

    # ------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------

    def next(self) -> int:
        """
        Sample the next bucket index and update internal state.
        Runtime is proportional to the number of active height levels.
        """
        k = len(self.bucket_count)
        if k <= 0:
            raise RuntimeError("no buckets available")

        # Fast path: beta == 0 reduces to IID uniform
        if self.beta == 0.0:
            b = self._rng.randrange(k)
            self._increment_bucket(b)
            return b

        total_weight = 0.0
        levels = []

        # Aggregate weights per height level
        base = self.min_count
        for c, lst in self.buckets_by_count.items():
            m = len(lst)
            delta = c - base
            w = m * self._exp_neg_beta(delta)
            levels.append((c, lst, m, w))
            total_weight += w

        # Sample a level
        r = self._rng.random() * total_weight
        acc = 0.0
        for c, lst, m, w in levels:
            acc += w
            if r <= acc:
                idx = self._rng.randrange(m)
                b = lst[idx]
                self._increment_bucket(
                    b,
                    known_count=c,
                    known_list=lst,
                    known_pos=idx,
                )
                return b

        # Numerical fallback
        c, lst, m, _ = levels[-1]
        b = lst[-1]
        self._increment_bucket(
            b,
            known_count=c,
            known_list=lst,
            known_pos=m - 1,
        )
        return b

    # ------------------------------------------------------------
    # Autoscaling helpers
    # ------------------------------------------------------------

    def add_buckets(self, n: int) -> None:
        """
        Add n new buckets.

        New buckets are initialized at the current minimum count so that
        their excess is zero (they start on the current 'floor').
        """
        if n <= 0:
            raise ValueError("n must be > 0")

        start_id = len(self.bucket_count)
        init_count = self.min_count
        level = self.buckets_by_count.setdefault(init_count, [])

        for b in range(start_id, start_id + n):
            self.bucket_count.append(init_count)
            self.bucket_pos.append(len(level))
            level.append(b)

    def remove_bucket(self, index: int) -> None:
        """
        Remove a bucket by index.

        This method is intentionally simple and may take O(k) time due to
        renumbering. Autoscaling down is assumed to be rare.
        """
        if index < 0 or index >= len(self.bucket_count):
            raise IndexError("bucket index out of range")
        if len(self.bucket_count) == 1:
            raise RuntimeError("cannot remove the last remaining bucket")

        c = self.bucket_count[index]
        pos = self.bucket_pos[index]
        self._remove_from_level(index, c, pos)

        del self.bucket_count[index]
        del self.bucket_pos[index]

        # Renumber bucket IDs > index
        for lst in self.buckets_by_count.values():
            for i, b in enumerate(lst):
                if b > index:
                    lst[i] = b - 1

        # Rebuild bucket_pos
        self.bucket_pos = [0] * len(self.bucket_count)
        for c, lst in self.buckets_by_count.items():
            for i, b in enumerate(lst):
                self.bucket_pos[b] = i

        self.min_count = min(self.buckets_by_count.keys())

    # ------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------

    def num_buckets(self) -> int:
        return len(self.bucket_count)

    def snapshot_counts(self) -> List[int]:
        return list(self.bucket_count)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _exp_neg_beta(self, delta: int) -> float:
        v = self._exp_cache.get(delta)
        if v is None:
            v = math.exp(-self.beta * delta)
            self._exp_cache[delta] = v
        return v

    def _increment_bucket(
        self,
        bucket: int,
        known_count: Optional[int] = None,
        known_list: Optional[List[int]] = None,
        known_pos: Optional[int] = None,
    ) -> None:
        """
        Move bucket from count c -> c+1 using swap-and-pop (O(1)).
        """
        c = known_count if known_count is not None else self.bucket_count[bucket]

        lst = known_list if known_list is not None else self.buckets_by_count[c]
        pos = known_pos if known_pos is not None else self.bucket_pos[bucket]

        # Remove from old level
        last = lst[-1]
        lst[pos] = last
        self.bucket_pos[last] = pos
        lst.pop()
        if not lst:
            del self.buckets_by_count[c]

        # Add to new level
        new_c = c + 1
        new_lst = self.buckets_by_count.setdefault(new_c, [])
        self.bucket_pos[bucket] = len(new_lst)
        new_lst.append(bucket)
        self.bucket_count[bucket] = new_c

        # Advance baseline lazily
        if c == self.min_count and c not in self.buckets_by_count:
            self.min_count += 1
            while self.min_count not in self.buckets_by_count:
                self.min_count += 1

    def _remove_from_level(self, bucket: int, c: int, pos: int) -> None:
        """
        Remove bucket from buckets_by_count[c] using swap-and-pop.
        """
        lst = self.buckets_by_count[c]
        last = lst[-1]
        lst[pos] = last
        self.bucket_pos[last] = pos
        lst.pop()
        if not lst:
            del self.buckets_by_count[c]
