import logging
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import eigsh
from miner.decorators.timer import timer
from sklearn.cluster import KMeans
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
        self._fiedler_vector: np.ndarray | None = None

        # Logger
        self.logger = logging.getLogger(__name__)

        # Cache
        self.latest_clusters: np.ndarray | None = None

    def remove_loops(self) -> csr_matrix:
        # Remove loops from the graph by subtracting the diagonal of the graph from itself
        no_loops_graph = self.graph - spdiags(
            self.graph.diagonal(), [0], self.graph.shape[0], self.graph.shape[1]
        )
        self.logger.debug(f"Removed loops from the graph: {no_loops_graph.shape}")
        self.graph = no_loops_graph
        return no_loops_graph

    def build_degree_matrix(self) -> np.ndarray:
        # D is the diagonal matrix who's (i, j) element is the sum of the i-th row of the graph
        degree_values = np.array(self.graph.sum(axis=1)).flatten()
        self.logger.debug(f"Computed degree values: {degree_values[:10]}...")
        return degree_values

    def build_laplacian(self) -> csr_matrix:
        # Build the Laplacian matrix L = D^(-1/2) * A * D^(-1/2)
        degree_values = self.build_degree_matrix()

        # Avoid division by zero
        degree_values = np.where(degree_values == 0, 1, degree_values)

        Dm1f2_values = 1 / np.sqrt(degree_values)
        Dm1f2 = spdiags(Dm1f2_values, [0], self.graph.shape[0], self.graph.shape[1])
        self.laplacian = Dm1f2 * self.graph * Dm1f2
        self.logger.debug(f"Computed Laplacian matrix: {self.laplacian.shape}")

        return self.laplacian

    @property
    def X(self) -> np.ndarray:
        if self.eigenvectors is None:
            self.compute_eigenvalues()
        return self.eigenvectors

    @timer(active=True)
    def compute_eigenvalues(self, k: int | None = None) -> EighResult:
        if k is None:
            k = self.k

        if self.laplacian is None:
            self.build_laplacian()

        # Use eigsh for sparse symmetric matrices (computes k smallest eigenvalues)
        # If k is None, compute all eigenvalues (up to n-1 for n×n matrix)

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

    def Y(self) -> np.ndarray:
        # We build the Y matrix by renormalizing each of X rows to have a unit length
        # Y_ij = X_ij / Sum_j(X^2_ij)^(1/2)

        # Avoid division by zero
        norm = np.linalg.norm(self.X, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)

        Y = self.X / norm
        return Y

    @property
    def fiedler_vector(self) -> np.ndarray:
        # Check if we already computed eigenvectors for clustering
        if self.eigenvectors is not None and self.eigenvectors.shape[1] >= 2:
            # The eigenvectors are already sorted descending (Largest to Smallest)
            # Index 0 is the stationary distribution (lambda ~ 1)
            # Index 1 is the Fiedler vector (lambda ~ 2nd largest)
            return self.eigenvectors[:, 1]

        # If not computed yet, compute the top 2 LARGEST
        if self.laplacian is None:
            self.build_laplacian()

        vals, vecs = eigsh(self.laplacian, k=2, which="LA")

        # Sort descending
        idx = vals.argsort()[::-1]
        sorted_vecs = vecs[:, idx]

        # Return 2nd vector
        if sorted_vecs.shape[1] >= 2:
            self._fiedler_vector = sorted_vecs[:, 1]
        else:
            self._fiedler_vector = sorted_vecs[:, 0]

        return self._fiedler_vector

    @fiedler_vector.setter
    def fiedler_vector(self, value: np.ndarray | None):
        """Setter for fiedler_vector property."""
        self._fiedler_vector = value

    @timer()
    def compute_kmeans(self) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.Y())
        return kmeans.predict(self.Y())

    def cluster(self):
        self.logger.debug("Starting clustering process")

        self.remove_loops()
        self.build_laplacian()
        self.compute_eigenvalues()

        clusters = self.compute_kmeans()
        self.latest_clusters = clusters

        return self.latest_clusters
