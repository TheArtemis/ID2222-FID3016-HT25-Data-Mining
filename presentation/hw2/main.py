from pathlib import Path
import sys
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.itemsets import test_apriori as test_apr
from test.itemsets import test_rule_generator as test_rules


def run_apriori() -> None:
    test_apr.test_apriori()


def run_rule_generator() -> None:
    test_rules.test_generator()


def run_selected(name: str) -> None:
    if name == "apriori":
        run_apriori()
    elif name == "rules":
        run_rule_generator()
    else:
        raise SystemExit(f"Unknown test: {name}. Choose from: apriori, rules")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="hw2-run", description="Run hw2 itemset demos"
    )
    parser.add_argument("test", nargs="?", choices=["apriori", "rules"], default=None)
    args = parser.parse_args()

    if args.test is None:
        prompt = "Select test (apriori, rules) [default: apriori]: "
        while True:
            try:
                ans = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                ans = "apriori"
            if ans == "":
                ans = "apriori"
            if ans in ("apriori", "rules"):
                args.test = ans
                break
            print(f"Invalid choice: {ans}. Choose one of: apriori, rules")

    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_selected(args.test)


if __name__ == "__main__":
    main()
