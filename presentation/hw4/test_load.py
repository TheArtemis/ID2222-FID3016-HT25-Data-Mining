import os
from pathlib import Path
import logging

from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"


def test_load_ex_1():
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    logger.info(f"Sparse matrix shape: {matrix.shape}")


def test_load_ex_2():
    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()
    logger.info(f"Sparse matrix shape: {matrix.shape}")


if __name__ == "__main__":
    if not EX_1_PATH.exists():
        raise FileNotFoundError(f"File {EX_1_PATH} not found")
    if not EX_2_PATH.exists():
        raise FileNotFoundError(f"File {EX_2_PATH} not found")

    test_load_ex_1()
    test_load_ex_2()
