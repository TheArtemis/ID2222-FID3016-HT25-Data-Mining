from pathlib import Path
import logging

from miner.core.spectra.gap_finder import GapFinder
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"


def test_ex_1():
    logger.info("\n" + "=" * 60)
    logger.info("Gap Analysis and Modularity for Example 1")
    logger.info("=" * 60 + "\n")

    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    gap_finder = GapFinder(matrix)

    # Find best k based on eigengap
    max_gap = gap_finder.find_best_k(max_k_to_check=15)
    logger.info(f"Best k based on eigengap: {max_gap}")

    # Analyze modularity for k=2 to k=10
    logger.info("\n" + "=" * 60)
    logger.info("Modularity Analysis")
    logger.info("=" * 60)
    modularity_scores = gap_finder.analyze_modularity(min_k=2, max_k=10)

    # Print results in a formatted table
    logger.info(f"\n{'k':<5} {'Modularity Q':<15}")
    logger.info("-" * 60)
    for k in sorted(modularity_scores.keys()):
        Q = modularity_scores[k]
        logger.info(f"{k:<5} {Q:<15.6f}")

    # Find best k based on modularity
    best_k_modularity = max(modularity_scores, key=modularity_scores.get)
    best_Q = modularity_scores[best_k_modularity]
    logger.info(f"\nBest k based on modularity: k={best_k_modularity} (Q={best_Q:.6f})")

    return {
        "best_k_gap": max_gap,
        "modularity_scores": modularity_scores,
        "best_k_modularity": best_k_modularity,
    }


def test_ex_2():
    logger.info("\n" + "=" * 60)
    logger.info("Gap Analysis and Modularity for Example 2")
    logger.info("=" * 60 + "\n")

    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()
    gap_finder = GapFinder(matrix)

    # Find best k based on eigengap
    max_gap = gap_finder.find_best_k(max_k_to_check=15)
    logger.info(f"Best k based on eigengap: {max_gap}")

    # Analyze modularity for k=2 to k=10
    logger.info("\n" + "=" * 60)
    logger.info("Modularity Analysis")
    logger.info("=" * 60)
    modularity_scores = gap_finder.analyze_modularity(min_k=2, max_k=10)

    # Print results in a formatted table
    logger.info(f"\n{'k':<5} {'Modularity Q':<15}")
    logger.info("-" * 60)
    for k in sorted(modularity_scores.keys()):
        Q = modularity_scores[k]
        logger.info(f"{k:<5} {Q:<15.6f}")

    # Find best k based on modularity
    best_k_modularity = max(modularity_scores, key=modularity_scores.get)
    best_Q = modularity_scores[best_k_modularity]
    logger.info(f"\nBest k based on modularity: k={best_k_modularity} (Q={best_Q:.6f})")

    return {
        "best_k_gap": max_gap,
        "modularity_scores": modularity_scores,
        "best_k_modularity": best_k_modularity,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_ex_1()
    test_ex_2()
