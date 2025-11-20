import logging
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import eigsh
from miner.decorators.timer import timer
import numpy as np
from miner.core.spectra.model import EighResult


class ClusterMachine:
    def __init__(self, graph: csr_matrix, k: int = 5):
        self.graph = graph  # Graph adjacency matrix (Represents the affinity matrix)
        self.k = k  # The k subsets we want to cluster the graph into

        # Store
        self.laplacian: csr_matrix | None = None
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None

        # Logger
        self.logger = logging.getLogger(__name__)

    def remove_loops(self) -> csr_matrix:
        # Remove loops from the graph by subtracting the diagonal of the graph from itself
        self.graph = self.graph - spdiags(
            self.graph.diagonal(), [0], self.graph.shape[0], self.graph.shape[1]
        )
        self.logger.debug(f"Removed loops from the graph: {self.graph.shape}")
        return self.graph

    def build_degree_matrix(self) -> np.ndarray:
        # D is the diagonal matrix who's (i, j) element is the sum of the i-th row of the graph
        degree_values = np.array(self.graph.sum(axis=1)).flatten()
        self.logger.debug(f"Computed degree values: {degree_values[:10]}...")
        return degree_values

    def build_laplacian(self) -> csr_matrix:
        # Build the Laplacian matrix L = D^(-1/2) * A * D^(-1/2)
        degree_values = self.build_degree_matrix()
        Dm1f2_values = 1 / np.sqrt(degree_values)
        Dm1f2 = spdiags(Dm1f2_values, [0], self.graph.shape[0], self.graph.shape[1])
        self.laplacian = Dm1f2 * self.graph * Dm1f2
        self.logger.debug(f"Computed Laplacian matrix: {self.laplacian.shape}")

        return self.laplacian

    @timer(active=True)
    def compute_eigenvalues(self, k: int | None = None) -> EighResult:
        # Use eigsh for sparse symmetric matrices (computes k smallest eigenvalues)
        # If k is None, compute all eigenvalues (up to n-1 for n×n matrix)
        if k is None:
            k = self.laplacian.shape[0] - 1

        # Compute the k largest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(self.laplacian, k=k, which="LA")

        # Sort the eigenvalues and eigenvectors in descending order
        # (x1, x2,..., xk)
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        self.logger.debug(f"Computed {k} largest eigenvalues and eigenvectors")

        return EighResult(eigenvalues=self.eigenvalues, eigenvectors=self.eigenvectors)

    def is_duplicate_eigenvalues(self) -> bool:
        result = all(self.eigenvalues == self.eigenvalues[0])
        self.logger.debug(f"Is duplicate eigenvalues: {result}")
        return result

    def cluster(self):
        self.logger.debug("Starting clustering process")
        self.remove_loops()
        self.build_laplacian()
        self.compute_eigenvalues()

        # This is not necessary as Laplacian matrix is symmetric (and thus it has no duplicate eigenvalues)
        duplicate_eigenvalues = self.is_duplicate_eigenvalues()
        if duplicate_eigenvalues:
            self.logger.warning("Duplicate eigenvalues found, clustering not possible")
            return

        pass
