from miner.core.itemsets.basket import Basket


class ItemsetAnalyzer:
    def __init__(self, baskets: list[Basket], s: float = 0.5, it: float = 0.5):
        self.baskets = baskets
        self.s = s  # Support threshold
        self.it = it  # Interest threshold

    def support(self, items: set[int]) -> float:
        if not self.baskets:
            return 0.0
        return sum(
            1 for basket in self.baskets if items.issubset(basket.itemset)
        ) / len(self.baskets)

    def is_frequent(self, items: set[int]) -> bool:
        return self.support(items) >= self.s

    def confidence(self, items: set[int], j: int) -> float:
        # support(I ∪ {j}) / support(I)
        if self.support(items) == 0:
            return 0.0
        return self.support(items | {j}) / self.support(items)

    def interest(self, items: set[int], j: int) -> float:
        return self.confidence(items, j) - self.support({j})
