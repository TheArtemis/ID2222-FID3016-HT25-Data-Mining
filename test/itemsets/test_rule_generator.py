import logging
from pathlib import Path

from miner.core.itemsets import Basket
from miner.core.itemsets.apriori import Apriori
from miner.core.itemsets.association_rules import AssociationRuleGenerator


ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "sales_transactions" / "T10I4D100K.dat"

# Experiment parameters
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.01
MIN_INTEREST = 0.01

logger = logging.getLogger(__name__)


def test_generator():
    baskets = Basket.load(DATASET_PATH)
    assert len(baskets) > 0

    apriori = Apriori(baskets, s=MIN_SUPPORT, it=MIN_INTEREST)
    apriori.run()

    frequent_table = apriori.frequent_items_table

    logger.info("Generating association rules...")
    rule_generator = AssociationRuleGenerator(
        frequent_itemsets=frequent_table,
        baskets=baskets,
        min_support=MIN_SUPPORT,
        min_confidence=MIN_CONFIDENCE,
        min_interest=MIN_INTEREST,
    )

    rules = rule_generator.generate()
    logger.info(f"Found {len(rules)} association rules")

    if rules:
        # Sort by interest (descending) to see most interesting rules first
        rules_sorted = sorted(rules, key=lambda r: r.interest, reverse=True)

        # Group rules by antecedent size
        rules_by_size = {}
        for rule in rules:
            size = len(rule.antecedent)
            if size not in rules_by_size:
                rules_by_size[size] = []
            rules_by_size[size].append(rule)

        logger.info("Rules:")
        for size in sorted(rules_by_size.keys()):
            logger.info(f"[Size {size}]: {len(rules_by_size[size])} rules")

        for rule in rules_sorted:
            logger.info(rule)

        avg_confidence = sum(r.confidence for r in rules) / len(rules)
        avg_interest = sum(r.interest for r in rules) / len(rules)
        logger.info("Rule statistics:")
        logger.info(f"  Average confidence: {avg_confidence:.4f}")
        logger.info(f"  Average interest: {avg_interest:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_generator()
