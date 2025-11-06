from __future__ import annotations
from functools import cached_property
import logging
from miner.core.itemsets import Basket, ItemsetAnalyzer


class FrequentItems:
    def __init__(self):
        self.itemsets: dict[frozenset[int], int] = {}

    @cached_property
    def max(self) -> int:
        if not self.itemsets:
            return 0
        return max(self.itemsets.values())

    def _invalidate_max_cache(self):
        if "max" in self.__dict__:
            delattr(self, "max")

    def add_itemset(self, itemset: frozenset[int], count: int):
        self.itemsets[itemset] = count
        self._invalidate_max_cache()

    def __repr__(self):
        return f"FrequentItems(count={len(self.itemsets)}, max={self.max})"


class Apriori:
    def __init__(self, baskets: list[Basket], s: float = 0.02, it: float = 0.1):
        self.logger = logging.getLogger(__name__)
        self.baskets = baskets
        self.s = s  # Support threshold
        self.it = it  # Interest threshold
        self.analyzer = ItemsetAnalyzer(baskets, s, it)
        self.frequent_items_table: dict[int, FrequentItems] = {}
        self.max_progressive = 1
        self.item_map: dict[int, int] = {}

    def first_pass(self):
        singletons: dict[frozenset[int], int] = {}
        for basket in self.baskets:
            for item in basket.itemset:
                # Build the item map -> key: original_item, value: progressive (1, 2, 3, ...)
                self.map_item(item)
                item_set = frozenset({self.item_map[item]})
                if item_set not in singletons:
                    singletons[item_set] = 1
                else:
                    singletons[item_set] += 1
        self.logger.debug(f"First pass: found {len(singletons)} unique singletons")
        if singletons:
            max_count = max(singletons.values())
            min_count = min(singletons.values())
            self.logger.debug(
                f"First pass: max count = {max_count}, min count = {min_count}"
            )
        return singletons

    def map_item(self, item: int):
        if item not in self.item_map:
            self.item_map[item] = self.max_progressive
            self.max_progressive = self.next_progressive()
            self.logger.debug(f"Mapped item {item} to {self.item_map[item]}")

    def next_progressive(self) -> int:
        return self.max_progressive + 1

    def generate_candidates(self, k: int, k_tons: dict[frozenset[int], int]):
        if k not in self.frequent_items_table:
            self.frequent_items_table[k] = FrequentItems()

        # Filter out items that are not frequent
        threshold = self.s * len(self.baskets)

        frequent_count = 0
        for itemset, count in k_tons.items():
            if count >= threshold:
                self.frequent_items_table[k].add_itemset(itemset, count)
                frequent_count += 1
            else:
                self.logger.debug(
                    f"Itemset {itemset} with count {count} below threshold {threshold}"
                )

        self.logger.debug(
            f"Level {k}: {frequent_count} itemsets out of {len(k_tons)} passed the threshold"
        )

    def process(self):
        singletons = self.first_pass()
        self.generate_candidates(1, singletons)
        # TODO continue here
