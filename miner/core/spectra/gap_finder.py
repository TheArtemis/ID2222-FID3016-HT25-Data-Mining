import logging
import numpy as np

from scipy.sparse import csr_matrix

from miner.core.spectra.cluster_machine import ClusterMachine


class GapFinder:
    def __init__(self, graph: csr_matrix):
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        self.cluster_machine = ClusterMachine(graph)
        self.gaps = None
        self.max_gap = None

    def find_best_k(self, max_k_to_check: int) -> int:
        """
        Finds the optimal number of clusters (k) by looking for the largest eigengap.
        Returns the integer k.
        """
        self.find_all_gaps(max_k_to_check)

        gap_index = np.argmax(self.gaps)

        best_k = gap_index + 1

        self.logger.info(
            f"Largest gap found at index {gap_index}, suggesting k={best_k}"
        )
        return best_k

    def find_all_gaps(self, max_k: int) -> list[float]:
        """
        Computes the first max_k eigenvalues and calculates the gaps.
        """
        self.cluster_machine.k = max_k

        self.cluster_machine.compute_eigenvalues()

        eigenvalues = self.cluster_machine.eigenvalues

        gaps = eigenvalues[:-1] - eigenvalues[1:]

        self.gaps = gaps

        if self.logger.isEnabledFor(logging.DEBUG):
            self.log_gaps()

        return gaps

    def log_gaps(self):
        for i, gap in enumerate(self.gaps):
            self.logger.debug(f"Gap between λ_{i + 1} and λ_{i + 2}: {gap:.4f}")
