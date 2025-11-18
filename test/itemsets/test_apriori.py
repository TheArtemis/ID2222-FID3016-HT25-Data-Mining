import logging
from pathlib import Path
import time

from miner.core.itemsets import Basket
from miner.core.itemsets.apriori import Apriori


ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "sales_transactions" / "T10I4D100K.dat"
logger = logging.getLogger(__name__)


MIN_SUPPORT = 0.001
MIN_INTEREST = 0.01


def test_apriori():
    baskets = Basket.load(DATASET_PATH)
    assert len(baskets) > 0
    logger.info(f"Loaded {len(baskets)} baskets")

    start_time = time.time()

    apriori = Apriori(baskets, s=MIN_SUPPORT, it=MIN_INTEREST)
    apriori.run()

    apriori_finish_time = time.time()

    logger.info("Frequent items table:")
    frequent_table = apriori.frequent_items_table
    logger.info(f"Total frequent itemsets: {len(frequent_table)}")

    logger.info(
        f"Itemsets by size: {[len(freq_collection.itemsets) for freq_collection in apriori.frequent_by_size]}"
    )

    # Show a few examples
    logger.info("Frequent itemsets:")
    for itemset, count in list(frequent_table.items()):
        logger.info(f"  {itemset}: {count} occurrences")

    logger.info(
        f"Time needed for the Apriori procedure: {apriori_finish_time - start_time:.3f} seconds"
    )


if __name__ == "__main__":
    test_apriori()
