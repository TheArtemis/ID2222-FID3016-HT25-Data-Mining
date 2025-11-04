from miner.core.itemsets.basket import Basket


class ItemsetAnalyzer:
    def __init__(self, baskets: list[Basket], s: float = 0.5):
        self.baskets = baskets
        self.s = s  # Support threshold

    def support(self, items: set[int]) -> float:
        if not self.baskets:
            return 0.0
        return sum(
            1 for basket in self.baskets if items.issubset(basket.itemset)
        ) / len(self.baskets)

    def is_frequent(self, items: set[int]) -> bool:
        return self.support(items) >= self.s
