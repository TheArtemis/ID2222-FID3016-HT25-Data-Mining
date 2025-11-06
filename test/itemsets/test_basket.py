import logging
from pathlib import Path

from miner.core.itemsets.basket import Basket


logger = logging.getLogger(__name__)

DATASET_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "sales_transactions"
    / "T10I4D100K.dat"
)


def test_basket():
    baskets = Basket.load(DATASET_PATH)
    logger.info(f"Number of baskets: {len(baskets)}")
    logger.info(f"First basket: {baskets[0]}")
    logger.info(f"Last basket: {baskets[-1]}")


if __name__ == "__main__":
    test_basket()
