from pathlib import Path
import logging
import numpy as np

from scipy.sparse import csr_matrix

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.cluster_plotter import ClusterPlotter
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EXAMPLE_PATH = DATA_DIR / "example1.dat"

TEST_DIR = ROOT_DIR / "test"
DUMP_DIR = TEST_DIR / "dumps"
PLOTS_DIR = DUMP_DIR / "plots"


MATRIX = csr_matrix(
    np.array(
        [
            [0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )
)


def ensure_paths():
    if not PLOTS_DIR.exists():
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_plots():
    # cluster_machine = ClusterMachine(MATRIX, k=4)
    graph_loader = GraphLoader(EXAMPLE_PATH)
    matrix = graph_loader.build(undirected=True)
    cluster_machine = ClusterMachine(matrix, k=10)
    cluster_machine.cluster()
    plotter = ClusterPlotter(cluster_machine)

    plotter.eigen_plotter.plot_sorted_eigenvalues(PLOTS_DIR, "matrix_eigenvalues.png")


if __name__ == "__main__":
    ensure_paths()
    test_plots()
