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

    def analyze_modularity(self, min_k: int = 2, max_k: int = 10) -> dict[int, float]:
        """
        Calculate modularity scores for different values of k (from min_k to max_k).
        
        This function clusters the graph for each k value and computes the modularity
        score to help corroborate eigengap findings.
        
        Args:
            min_k: Minimum number of clusters to test (default: 2)
            max_k: Maximum number of clusters to test (default: 10)
        
        Returns:
            Dictionary mapping k values to their modularity scores
        """
        modularity_scores: dict[int, float] = {}
        
        self.logger.info(f"Analyzing modularity for k={min_k} to k={max_k}")
        
        for k in range(min_k, max_k + 1):
            # Create a new ClusterMachine for this k value
            # We need to ensure the graph is in the right state (no loops)
            temp_machine = ClusterMachine(self.graph, k=k)
            
            # Perform clustering
            clusters = temp_machine.cluster()
            
            # Calculate modularity
            Q = ClusterMachine.calculate_modularity(self.graph, clusters)
            modularity_scores[k] = Q
            
            self.logger.info(f"k={k}: Modularity Q = {Q:.6f}")
        
        return modularity_scores
