from pathlib import Path
import logging

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"


def test_cluster_machine():
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix)

    cluster_machine.cluster()
    logger.info("Cluster machine successfully clustered")


if __name__ == "__main__":
    test_cluster_machine()
