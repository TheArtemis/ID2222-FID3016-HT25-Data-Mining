from __future__ import annotations
import logging
from collections.abc import Mapping
from miner.core.itemsets import Basket, ItemsetAnalyzer


class AssociationRule:
    def __init__(
        self,
        antecedent: frozenset[int],
        consequent: int,
        support: float,
        confidence: float,
        interest: float,
    ):
        self.antecedent = antecedent
        self.consequent = consequent
        self.support = support
        self.confidence = confidence
        self.interest = interest

    def __repr__(self) -> str:
        antecedent_str = (
            "{" + ", ".join(sorted(str(x) for x in self.antecedent)) + "}"
            if self.antecedent
            else "{}"
        )
        return (
            f"AssociationRule({antecedent_str} -> {self.consequent}, "
            f"s={self.support:.4f}, c={self.confidence:.4f}, "
            f"i={self.interest:.4f})"
        )


class AssociationRuleGenerator:
    """Generates association rules from frequent itemsets."""

    def __init__(
        self,
        frequent_itemsets: Mapping[tuple[int, ...], int],
        baskets: list[Basket],
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_interest: float = 0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.frequent_itemsets = frequent_itemsets
        self.baskets = baskets
        self.analyzer = ItemsetAnalyzer(baskets, s=min_support, it=min_interest)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_interest = min_interest

    def generate(self) -> list[AssociationRule]:
        rules = []

        # Only generate rules from itemsets of size >= 2
        for itemset, _ in self.frequent_itemsets.items():
            self.logger.debug(f"Processing itemset: {itemset}")
            if len(itemset) < 2:
                continue

            rules.extend(self.process_itemset(itemset))

        self.logger.debug(
            f"Generated {len(rules)} association rules "
            f"(support >= {self.min_support}, confidence >= {self.min_confidence}, "
            f"interest >= {self.min_interest})"
        )
        return rules

    def process_itemset(self, itemset: tuple[int, ...]) -> list[AssociationRule]:
        rules = []
        items = list(itemset)

        for index, j in enumerate(items):
            rule = self.generate_rule(items, index, j)

            # Check if the rule meets the thresholds
            if self.filter_rule(rule):
                self.logger.debug(f"Adding rule: {rule}")
                rules.append(rule)

        return rules

    def filter_rule(self, rule: AssociationRule) -> bool:
        return (
            rule.support >= self.min_support
            and rule.confidence >= self.min_confidence
            and rule.interest >= self.min_interest
        )

    def generate_rule(self, items: list[int], index: int, j: int) -> AssociationRule:
        antecedent_items = tuple(items[:index] + items[index + 1 :])
        antecedent = frozenset(antecedent_items)
        consequent = j

        antecedent_set = set(antecedent_items)
        itemset_support = self.analyzer.support(set(items))

        confidence = self.analyzer.confidence(antecedent_set, consequent)
        interest = self.analyzer.interest(antecedent_set, consequent)

        return AssociationRule(
            antecedent=antecedent,
            consequent=consequent,
            support=itemset_support,
            confidence=confidence,
            interest=interest,
        )
