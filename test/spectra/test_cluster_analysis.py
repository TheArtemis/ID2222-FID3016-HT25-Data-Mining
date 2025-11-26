from pathlib import Path
import logging

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"
PRESENTATION_DIR = ROOT_DIR / "presentation" / "hw4" / "imgs"
EXAMPLE_1_FOLDER = PRESENTATION_DIR / "example1"
EXAMPLE_2_FOLDER = PRESENTATION_DIR / "example2"
K1 = 4
K2 = 2

EXAMPLE_1_K_FOLDER = EXAMPLE_1_FOLDER / f"K_{K1}"
EXAMPLE_2_K_FOLDER = EXAMPLE_2_FOLDER / f"K_{K2}"


def ensure_paths():
    if not EXAMPLE_1_K_FOLDER.exists():
        EXAMPLE_1_K_FOLDER.mkdir(parents=True, exist_ok=True)
    if not EXAMPLE_2_K_FOLDER.exists():
        EXAMPLE_2_K_FOLDER.mkdir(parents=True, exist_ok=True)


def test_cluster_analysis_1():
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K1)
    cluster_machine.cluster()

    # Internal connectivity

    logger.info(f"\n{'-' * 50}\nInternal connectivity\n{'-' * 50}")
    intra_cluster_edges_count = cluster_machine.get_total_intra_cluster_edges_count()
    logger.info(f"Intra-cluster edges count: {intra_cluster_edges_count}")

    # External connectivity

    logger.info(f"\n{'-' * 50}\nExternal connectivity\n{'-' * 50}")
    inter_cluster_edges_count = cluster_machine.get_total_inter_cluster_edges_count()
    logger.info(f"Inter-cluster edges count: {inter_cluster_edges_count}")

    expansion_ratio = cluster_machine.get_expansion_ratio()
    logger.info(f"Expansion ratio: {expansion_ratio}")

    conductance = cluster_machine.get_conductance()
    logger.info(f"Conductance: {conductance}")


def test_cluster_analysis_2():
    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K2)
    cluster_machine.cluster()

    # Internal connectivity
    logger.info(f"\n{'-' * 50}\nInternal connectivity\n{'-' * 50}")
    intra_cluster_edges_count = cluster_machine.get_total_intra_cluster_edges_count()
    logger.info(f"Intra-cluster edges count: {intra_cluster_edges_count}")

    # External connectivity
    logger.info(f"\n{'-' * 50}\nExternal connectivity\n{'-' * 50}")
    inter_cluster_edges_count = cluster_machine.get_total_inter_cluster_edges_count()
    logger.info(f"Inter-cluster edges count: {inter_cluster_edges_count}")

    expansion_ratio = cluster_machine.get_expansion_ratio()
    logger.info(f"Expansion ratio: {expansion_ratio}")

    conductance = cluster_machine.get_conductance()
    logger.info(f"Conductance: {conductance}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"\n{'-' * 50}\nCluster analysis for example 1\n{'-' * 50}")
    test_cluster_analysis_1()
    logger.info(f"\n{'-' * 50}\nCluster analysis for example 2\n{'-' * 50}")
    test_cluster_analysis_2()
