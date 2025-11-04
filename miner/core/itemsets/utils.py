from miner.core.itemsets.basket import Basket


def support(items: set[int], baskets: list[Basket]) -> int:
    return sum(1 for basket in baskets if items.issubset(basket.itemset)) / len(baskets)


def is_frequent(items: set[int], baskets: list[Basket], s=0.5) -> bool:
    return support(items, baskets) >= s
