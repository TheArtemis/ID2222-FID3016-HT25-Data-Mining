from __future__ import annotations
from functools import cached_property
from itertools import combinations
import logging
from miner.core.itemsets import Basket, ItemsetAnalyzer


class FrequentSetsCollection:
    def __init__(self):
        self.itemsets: dict[tuple[int, ...], int] = {}

    @cached_property
    def max(self) -> int:
        return max(self.itemsets.values()) if self.itemsets else 0

    def _invalidate_max_cache(self):
        self.__dict__.pop("max", None)

    def add(self, itemset: tuple[int, ...], count: int):
        self.itemsets[itemset] = count
        self._invalidate_max_cache()

    def __repr__(self):
        return f"FrequentItems(count={len(self.itemsets)}, max={self.max})"


class Apriori:
    def __init__(
        self,
        baskets: list[Basket],
        s: float = 0.01,
        it: float = 0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.baskets = baskets
        self.s = s  # Support threshold
        self.it = it  # Interest threshold
        self.analyzer = ItemsetAnalyzer(baskets, s, it)
        self.frequent_by_size: list[FrequentSetsCollection] = []
        self.threshold = s * len(baskets)

    def run(self):
        """Main method to process the Apriori algorithm."""
        k = 1
        while True:
            self.logger.debug(f"Starting pass {k}")
            frequent = self.run_pass(k)

            if not frequent:
                self.logger.debug(f"No frequent itemsets at pass {k}, stopping")
                break

            collection = FrequentSetsCollection()
            for itemset, count in frequent.items():
                collection.add(itemset, count)
            self.frequent_by_size.append(collection)
            self.logger.debug(f"Pass {k}: found {len(frequent)} frequent {k}-itemsets")
            k += 1

        self.logger.info(
            f"Complete: found frequent itemsets up to size {len(self.frequent_by_size)}"
        )

    def count_singletons(self) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = {}
        for basket in self.baskets:
            for item in basket.itemset:
                singleton = (item,)
                counts[singleton] = counts.get(singleton, 0) + 1
        return counts

    def join_itemsets(
        self, s1: tuple[int, ...], s2: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        # Check a < b to avoid duplicates
        if s1[:-1] == s2[:-1] and s1[-1] < s2[-1]:
            return s1 + (s2[-1],)
        return None

    def has_frequent_subsets(
        self, candidate: tuple[int, ...], prev_frequent: set[tuple[int, ...]]
    ) -> bool:
        subset_size = len(candidate) - 1
        for subset in combinations(candidate, subset_size):
            # If the subset is not in the previous frequent itemsets, the candidate it's not frequent
            # ex. {1, 2, 3} is not frequent if {1, 3} is not frequent
            if subset not in prev_frequent:
                return False
        return True

    def generate_candidates(
        self, prev_frequent: dict[tuple[int, ...], int]
    ) -> set[tuple[int, ...]]:
        candidates = set()

        # Get the keys of the previous frequent itemsets: [{1, 2, 3}, {1, 2, 4}, {1, 2, 5}, ...]
        itemsets = list(prev_frequent.keys())

        # Faster lookup with a set
        prev_set = set(itemsets)

        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Joins the two sets into a new one: {1, 2, 3} + {1, 2, 4} = {1, 2, 3, 4}
                candidate = self.join_itemsets(itemsets[i], itemsets[j])

                # Check if the candidate has frequent subsets in the previous pass
                if candidate and self.has_frequent_subsets(candidate, prev_set):
                    candidates.add(candidate)

        return candidates

    def count_candidates(
        self, candidates: set[tuple[int, ...]], k: int
    ) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = {}

        for basket in self.baskets:
            items = sorted(basket.itemset)
            if len(items) < k:
                continue

            for combo in combinations(items, k):
                if combo in candidates:
                    counts[combo] = counts.get(combo, 0) + 1

        return counts

    def filter_frequent(
        self, counts: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, ...], int]:
        return {
            itemset: count
            for itemset, count in counts.items()
            if count >= self.threshold
        }

    def run_pass(self, k: int) -> dict[tuple[int, ...], int]:
        if k == 1:
            counts = self.count_singletons()
        else:
            if not self.frequent_by_size:
                return {}

            # Get the itemsets of the previous pass
            previous_itemsets: dict[tuple[int, ...], int] = (
                self.get_previous_frequent_itemsets(k - 1)
            )
            if not previous_itemsets:
                return {}

            # Generate candidates for the current pass
            candidates: set[tuple[int, ...]] = self.generate_candidates(
                previous_itemsets
            )
            if not candidates:
                return {}

            # Count the candidates
            counts: dict[tuple[int, ...], int] = self.count_candidates(candidates, k)

        frequent: dict[tuple[int, ...], int] = self.filter_frequent(counts)

        self.logger.debug(
            f"Pass {k}: {len(counts)} candidates  {len(frequent)} frequent (threshold={self.threshold:.1f})"
        )

        return frequent

    def get_previous_frequent_itemsets(self, k: int) -> dict[tuple[int, ...], int]:
        if 0 < k <= len(self.frequent_by_size):
            return self.frequent_by_size[k - 1].itemsets
        return {}

    def get_frequent(self, size: int) -> dict[tuple[int, ...], int]:
        if 0 < size <= len(self.frequent_by_size):
            return self.frequent_by_size[size - 1].itemsets
        return {}

    def all_frequent_itemsets(self) -> list[tuple[tuple[int, ...], int]]:
        result = []
        for frequent_collection in self.frequent_by_size:
            result.extend(frequent_collection.itemsets.items())
        return result

    @property
    def frequent_items_table(self) -> dict[tuple[int, ...], int]:
        """Returns a dictionary of all frequent itemsets with their counts."""
        result = {}
        for frequent_collection in self.frequent_by_size:
            result.update(frequent_collection.itemsets)
        return result
