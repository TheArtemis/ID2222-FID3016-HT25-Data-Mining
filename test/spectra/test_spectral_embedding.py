from pathlib import Path
import logging

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.cluster_plotter import ClusterPlotter
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
PRESENTATION_DIR = ROOT_DIR / "presentation" / "hw4" / "imgs"
EXAMPLE_1_FOLDER = PRESENTATION_DIR / "example1"


def get_k_folder(k: int) -> Path:
    """Get the output folder for a specific k value."""
    return EXAMPLE_1_FOLDER / f"K_{k}"


def ensure_paths(k: int = 4):
    """Ensure output directories exist."""
    k_folder = get_k_folder(k)
    if not k_folder.exists():
        k_folder.mkdir(parents=True, exist_ok=True)


def test_spectral_embedding_example1(k: int = 4):
    """
    Test spectral embedding visualization for example1.
    Creates a 2D scatter plot of the 2nd vs 3rd eigenvectors colored by cluster.

    Args:
        k: Number of clusters to use for the visualization (default: 4)
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"Spectral Embedding Visualization for Example 1 (k={k})")
    logger.info("=" * 60 + "\n")

    # Ensure output directory exists
    k_folder = get_k_folder(k)
    ensure_paths(k)

    # Load graph
    graph_loader = GraphLoader(EX_1_PATH)
    matrix = graph_loader.build()
    logger.info(f"Loaded graph with {matrix.shape[0]} nodes")

    # Create ClusterMachine and perform clustering
    cluster_machine = ClusterMachine(matrix, k=k)
    clusters = cluster_machine.cluster()
    logger.info(f"Clustering completed. Cluster assignments: {clusters}")

    # Create plotter and generate spectral embedding plot
    plotter = ClusterPlotter(cluster_machine)
    plotter.plot_spectral_embedding(
        k_folder,
        filename=f"example1_spectral_embedding_k_{k}.png",
        figsize=(10, 8),
    )
    logger.info(f"Spectral embedding plot saved to {k_folder}")

    return cluster_machine, plotter


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_spectral_embedding_example1(k=4)
