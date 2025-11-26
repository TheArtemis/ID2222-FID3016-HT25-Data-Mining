from pathlib import Path
import logging

from pydantic import BaseModel

from miner.core.spectra.cluster_machine import ClusterMachine
from miner.core.spectra.graph_loader import GraphLoader

logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "graph"
EX_1_PATH = DATA_DIR / "example1.dat"
EX_2_PATH = DATA_DIR / "example2.dat"


K_RANGE = range(2, 10)


class K_Value(BaseModel):
    k: int
    value: float | int | None


class ClusterGoodnessResults(BaseModel):
    best_inter_cluster_edges_count: K_Value | None = None
    best_intra_cluster_edges_count: K_Value | None = None
    best_expansion_ratio: K_Value | None = None
    best_conductance: K_Value | None = None


def test_cluster_goodness_1():
    graph_loader = GraphLoader(EX_2_PATH)
    matrix = graph_loader.build()

    results = ClusterGoodnessResults()

    for k in K_RANGE:
        cluster_machine = ClusterMachine(matrix, k=k)
        cluster_machine.cluster()
        inter_cluster_edges_count = (
            cluster_machine.get_total_inter_cluster_edges_count()
        )
        intra_cluster_edges_count_dict = (
            cluster_machine.get_total_intra_cluster_edges_count()
        )
        # Sum all intra-cluster edges across all clusters
        intra_cluster_edges_count = sum(intra_cluster_edges_count_dict.values())
        logger.info(f"K: {k}")
        logger.info(f"Inter-cluster edges count: {inter_cluster_edges_count}")
        logger.info(f"Intra-cluster edges count: {intra_cluster_edges_count}")
        expansion_ratio_result = cluster_machine.get_expansion_ratio()
        conductance_result = cluster_machine.get_conductance()
        logger.info(f"Expansion ratio: {expansion_ratio_result}")
        logger.info(f"Conductance: {conductance_result}")
        logger.info("--------------------------------")

        # Calculate average expansion ratio and conductance across all clusters
        expansion_ratio_values = list(expansion_ratio_result.data.values())
        # Filter out infinity values for averaging
        expansion_ratio_finite = [
            v for v in expansion_ratio_values if v != float("inf")
        ]
        avg_expansion_ratio = (
            sum(expansion_ratio_finite) / len(expansion_ratio_finite)
            if expansion_ratio_finite
            else float("inf")
        )

        conductance_values = list(conductance_result.data.values())
        avg_conductance = (
            sum(conductance_values) / len(conductance_values)
            if conductance_values
            else 0.0
        )

        if (
            results.best_inter_cluster_edges_count is None
            or inter_cluster_edges_count <= results.best_inter_cluster_edges_count.value
        ):
            results.best_inter_cluster_edges_count = K_Value(
                k=k, value=inter_cluster_edges_count
            )
        if (
            results.best_intra_cluster_edges_count is None
            or intra_cluster_edges_count >= results.best_intra_cluster_edges_count.value
        ):
            results.best_intra_cluster_edges_count = K_Value(
                k=k, value=intra_cluster_edges_count
            )
        if results.best_expansion_ratio is None:
            results.best_expansion_ratio = K_Value(k=k, value=avg_expansion_ratio)
        elif avg_expansion_ratio != float("inf") and (
            results.best_expansion_ratio.value == float("inf")
            or avg_expansion_ratio <= results.best_expansion_ratio.value
        ):
            results.best_expansion_ratio = K_Value(k=k, value=avg_expansion_ratio)
        if (
            results.best_conductance is None
            or avg_conductance <= results.best_conductance.value
        ):
            results.best_conductance = K_Value(k=k, value=avg_conductance)

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    test_cluster_goodness_1()
