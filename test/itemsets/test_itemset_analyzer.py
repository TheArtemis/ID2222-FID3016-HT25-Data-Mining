import logging
from miner.core.itemsets.basket import Basket
from miner.core.itemsets.itemset_analyzer import ItemsetAnalyzer

logger = logging.getLogger(__name__)

baskets = [
    Basket(itemset={1, 2, 3}),
    Basket(itemset={1, 2, 4}),
    Basket(itemset={1, 3, 4}),
    Basket(itemset={2, 3, 4}),
]

items = [{1, 2}, {1, 3}, {2, 3}, {1, 2, 3}, {1, 2, 3, 4}]


def test_support():
    analyzer = ItemsetAnalyzer(baskets)
    assert analyzer.support(items[0]) == 0.5
    assert analyzer.support(items[1]) == 0.5
    assert analyzer.support(items[2]) == 0.5
    assert analyzer.support(items[3]) == 0.25
    assert analyzer.support(items[4]) == 0.0
    logger.info(f"Support of {items[0]}: {analyzer.support(items[0])}")
    logger.info(f"Support of {items[1]}: {analyzer.support(items[1])}")
    logger.info(f"Support of {items[2]}: {analyzer.support(items[2])}")
    logger.info(f"Support of {items[3]}: {analyzer.support(items[3])}")
    logger.info(f"Support of {items[4]}: {analyzer.support(items[4])}")


def test_is_frequent():
    analyzer = ItemsetAnalyzer(baskets, s=0.5)
    assert analyzer.is_frequent(items[0])
    assert analyzer.is_frequent(items[1])
    assert analyzer.is_frequent(items[2])
    assert not analyzer.is_frequent(items[3])
    assert not analyzer.is_frequent(items[4])

    logger.info(f"Is frequent {items[0]}: {analyzer.is_frequent(items[0])}")
    logger.info(f"Is frequent {items[1]}: {analyzer.is_frequent(items[1])}")
    logger.info(f"Is frequent {items[2]}: {analyzer.is_frequent(items[2])}")
    logger.info(f"Is frequent {items[3]}: {analyzer.is_frequent(items[3])}")
    logger.info(f"Is frequent {items[4]}: {analyzer.is_frequent(items[4])}")


def test_confidence():
    analyzer = ItemsetAnalyzer(baskets)
    logger.info(f"Confidence of {items[0]}: {analyzer.confidence(items[0], 3)}")
    logger.info(f"Confidence of {items[1]}: {analyzer.confidence(items[1], 4)}")
    logger.info(f"Confidence of {items[2]}: {analyzer.confidence(items[2], 3)}")
    logger.info(f"Confidence of {items[3]}: {analyzer.confidence(items[3], 4)}")
    logger.info(f"Confidence of {items[4]}: {analyzer.confidence(items[4], 3)}")


def test_interest():
    analyzer = ItemsetAnalyzer(baskets)

    logger.info(f"Interest of {items[0]}: {analyzer.interest(items[0], 3)}")
    logger.info(f"Interest of {items[1]}: {analyzer.interest(items[1], 4)}")
    logger.info(f"Interest of {items[2]}: {analyzer.interest(items[2], 3)}")
    logger.info(f"Interest of {items[3]}: {analyzer.interest(items[3], 4)}")
    logger.info(f"Interest of {items[4]}: {analyzer.interest(items[4], 3)}")


if __name__ == "__main__":
    test_support()
    logger.info("--------------------------------")
    test_is_frequent()
    logger.info("--------------------------------")
    test_confidence()
    logger.info("--------------------------------")
    test_interest()
    logger.info("--------------------------------")
