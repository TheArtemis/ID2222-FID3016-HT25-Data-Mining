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
K1 = 4
K2 = 2

EXAMPLE_1_K_FOLDER = EXAMPLE_1_FOLDER / f"K_{K1}"
EXAMPLE_2_K_FOLDER = EXAMPLE_2_FOLDER / f"K_{K2}"


def ensure_paths():
    if not EXAMPLE_1_K_FOLDER.exists():
        EXAMPLE_1_K_FOLDER.mkdir(parents=True, exist_ok=True)
    if not EXAMPLE_2_K_FOLDER.exists():
        EXAMPLE_2_K_FOLDER.mkdir(parents=True, exist_ok=True)


def test_cluster_machine_1(K: int):
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K)

    cluster_machine.cluster()
    logger.info(cluster_machine.latest_clusters)

    plotter = ClusterPlotter(cluster_machine)
    plotter.plot_clusters(EXAMPLE_1_K_FOLDER, f"example1_clusters_K_{K}.png")

    plotter.plot_graph_structure(
        EXAMPLE_1_K_FOLDER,
        f"example1_graph_structure_K_{K}.png",
        layout="spring",
        node_size=70,
    )
    plotter.plot_fiedler_vector(
        EXAMPLE_1_K_FOLDER, f"example1_fiedler_vector_K_{K}.png"
    )
    plotter.plot_fiedler_vector_on_graph(
        EXAMPLE_1_K_FOLDER,
        f"example1_fiedler_vector_graph_K_{K}.png",
        layout="spring",
        node_size=70,
    )


def test_cluster_machine_2(K: int):
    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()
    cluster_machine = ClusterMachine(matrix, k=K)

    cluster_machine.cluster()
    logger.info(cluster_machine.latest_clusters)

    plotter = ClusterPlotter(cluster_machine)
    plotter.plot_clusters(EXAMPLE_2_K_FOLDER, f"example2_clusters_K_{K}.png")

    plotter.plot_graph_structure(
        EXAMPLE_2_K_FOLDER,
        f"example2_graph_structure_K_{K}.png",
        layout="spring",
        node_size=70,
    )
    plotter.plot_fiedler_vector(
        EXAMPLE_2_K_FOLDER, f"example2_fiedler_vector_K_{K}.png"
    )
    plotter.plot_fiedler_vector_on_graph(
        EXAMPLE_2_K_FOLDER,
        f"example2_fiedler_vector_graph_K_{K}.png",
        layout="circular",
        node_size=70,
    )


if __name__ == "__main__":
    ensure_paths()
    test_cluster_machine_1(K1)
    test_cluster_machine_2(K2)
