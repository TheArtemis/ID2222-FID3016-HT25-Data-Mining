from __future__ import annotations
from miner.core.itemsets import Basket, ItemsetAnalyzer


class FrequentItems:
    def __init__(self):
        self.itemsets: dict[frozenset[int], int] = {}


class Apriori:
    def __init__(self, baskets: list[Basket], s: float = 0.5, it: float = 0.0):
        self.baskets = baskets
        self.s = s  # Support threshold
        self.it = it  # Interest threshold
        self.analyzer = ItemsetAnalyzer(baskets, s, it)
        self.frequent_items_table: dict[int, FrequentItems] = {}

    def first_pass(self):
        singletons: dict[frozenset[int], int] = {}
        for basket in self.baskets:
            for item in basket.itemset:
                item_set = frozenset({item})
                if item_set not in singletons:
                    singletons[item_set] = 1
                else:
                    singletons[item_set] += 1
        return singletons

    def generate_candidates(self, k: int, k_tons: dict[frozenset[int], int]):
        if k not in self.frequent_items_table:
            self.frequent_items_table[k] = FrequentItems()

        # Filter out items that are not frequent
        threshold = self.s * len(self.baskets)
        for itemset, count in k_tons.items():
            if count >= threshold:
                self.frequent_items_table[k].itemsets[itemset] = count

    def process(self):
        k = 1
        singletons = self.first_pass()
        self.generate_candidates(1, singletons)
        # TODO continue here
