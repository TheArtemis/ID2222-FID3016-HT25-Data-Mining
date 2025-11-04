import logging
from miner.core.itemsets.basket import Basket
from miner.core.itemsets.utils import is_frequent, support

logger = logging.getLogger(__name__)

baskets = [
    Basket(itemset={1, 2, 3}),
    Basket(itemset={1, 2, 4}),
    Basket(itemset={1, 3, 4}),
    Basket(itemset={2, 3, 4}),
]

items = [{1, 2}, {1, 3}, {2, 3}, {1, 2, 3}, {1, 2, 3, 4}]


def test_support():
    assert support(items[0], baskets) == 0.5
    assert support(items[1], baskets) == 0.5
    assert support(items[2], baskets) == 0.5
    assert support(items[3], baskets) == 0.25
    assert support(items[4], baskets) == 0.0
    logger.info(f"Support of {items[0]}: {support(items[0], baskets)}")
    logger.info(f"Support of {items[1]}: {support(items[1], baskets)}")
    logger.info(f"Support of {items[2]}: {support(items[2], baskets)}")
    logger.info(f"Support of {items[3]}: {support(items[3], baskets)}")
    logger.info(f"Support of {items[4]}: {support(items[4], baskets)}")


def test_is_frequent():
    assert is_frequent(items[0], baskets)
    assert is_frequent(items[1], baskets)
    assert is_frequent(items[2], baskets)
    assert not is_frequent(items[3], baskets)
    assert not is_frequent(items[4], baskets)

    logger.info(f"Is frequent {items[0]}: {is_frequent(items[0], baskets)}")
    logger.info(f"Is frequent {items[1]}: {is_frequent(items[1], baskets)}")
    logger.info(f"Is frequent {items[2]}: {is_frequent(items[2], baskets)}")
    logger.info(f"Is frequent {items[3]}: {is_frequent(items[3], baskets)}")
    logger.info(f"Is frequent {items[4]}: {is_frequent(items[4], baskets)}")


if __name__ == "__main__":
    test_support()
    logger.info("--------------------------------")
    test_is_frequent()
    logger.info("--------------------------------")
