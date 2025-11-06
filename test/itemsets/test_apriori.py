from pathlib import Path

from miner.core.itemsets import Basket
from miner.core.itemsets.apriori import Apriori


ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "sales_transactions" / "T10I4D100K.dat"


def test_apriori():
    baskets = Basket.load(DATASET_PATH)
    assert len(baskets) > 0
    print(f"Loaded {len(baskets)} baskets")

    apriori = Apriori(baskets)
    apriori.process()

    print("Frequent items table:")
    print(apriori.frequent_items_table)


if __name__ == "__main__":
    test_apriori()
