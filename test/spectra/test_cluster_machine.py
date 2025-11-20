from pathlib import Path
import logging

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.cluster_plotter import ClusterPlotter
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"
PRESENTATION_DIR = ROOT_DIR / "presentation" / "hw4" / "imgs"
EXAMPLE_1_FOLDER = PRESENTATION_DIR / "example1"
EXAMPLE_2_FOLDER = PRESENTATION_DIR / "example2"

if not EXAMPLE_1_FOLDER.exists():
    EXAMPLE_1_FOLDER.mkdir(parents=True, exist_ok=True)
if not EXAMPLE_2_FOLDER.exists():
    EXAMPLE_2_FOLDER.mkdir(parents=True, exist_ok=True)

K = 10


def test_cluster_machine_1():
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K)

    cluster_machine.cluster()
    logger.info(cluster_machine.latest_clusters)

    plotter = ClusterPlotter(cluster_machine)
    plotter.plot_clusters(EXAMPLE_1_FOLDER, f"example1_clusters_K_{K}.png")
    plotter.plot_eigenvalue_spectrum(
        EXAMPLE_1_FOLDER, f"example1_eigenvalue_spectrum_K_{K}.png"
    )


def test_cluster_machine_2():
    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K)

    cluster_machine.cluster()
    logger.info(cluster_machine.latest_clusters)

    plotter = ClusterPlotter(cluster_machine)
    plotter.plot_clusters(EXAMPLE_2_FOLDER, f"example2_clusters_K_{K}.png")
    plotter.plot_eigenvalue_spectrum(
        EXAMPLE_2_FOLDER, f"example2_eigenvalue_spectrum_K_{K}.png"
    )


if __name__ == "__main__":
    test_cluster_machine_1()
    test_cluster_machine_2()
