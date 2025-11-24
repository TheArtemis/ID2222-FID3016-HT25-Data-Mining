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
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    gap_finder = GapFinder(matrix)
    max_gap = gap_finder.find_best_k(max_k_to_check=15)
    logger.info(f"Best k: {max_gap}")


if __name__ == "__main__":
    test_ex_1()
