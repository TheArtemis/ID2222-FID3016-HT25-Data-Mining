import logging
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import eigsh
from miner.decorators.timer import timer
from sklearn.cluster import KMeans
import numpy as np
from miner.core.spectra.model import EighResult, ClusterAnalysisResult


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
        # Use fixed random seed for deterministic results
        np.random.seed(0)
        eigenvalues, eigenvectors = eigsh(
            self.laplacian,
            k=k,
            which="LA",
            v0=np.random.RandomState(0).randn(
                self.laplacian.shape[0]
            ),  # Fixed initial vector
        )

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

    def _get_deterministic_initial_centroids(self, Y: np.ndarray) -> np.ndarray:
        """
        Generate deterministic initial centroids for k-means clustering.

        Selects points at regular intervals in the data to ensure reproducible
        initialization regardless of random state.

        Args:
            Y: The normalized eigenvector matrix (n_samples, n_features)

        Returns:
            Array of initial centroids (k, n_features)
        """
        n_samples = Y.shape[0]
        if n_samples >= self.k:
            # Select indices at regular intervals
            indices = np.linspace(0, n_samples - 1, self.k, dtype=int)
            init_centroids = Y[indices].copy()
        else:
            # If we have fewer samples than clusters, use all samples and pad
            init_centroids = Y.copy()
            # Pad with first sample repeated if needed (shouldn't happen in practice)
            while init_centroids.shape[0] < self.k:
                init_centroids = np.vstack([init_centroids, Y[0:1]])

        return init_centroids

    @timer()
    def compute_kmeans(self) -> np.ndarray:
        """
        Perform k-means clustering on the normalized eigenvector matrix.

        Uses deterministic initialization to ensure reproducible results.

        Returns:
            Cluster assignments for each node (n_samples,)
        """
        Y = self.Y()
        init_centroids = self._get_deterministic_initial_centroids(Y)

        kmeans = KMeans(
            n_clusters=self.k,
            random_state=0,
            n_init=1,  # Single initialization for determinism
            init=init_centroids,  # Use fixed initial centroids
            max_iter=300,  # Explicit max iterations
            algorithm="lloyd",  # Explicit algorithm
            tol=1e-4,  # Explicit tolerance
        ).fit(Y)
        return kmeans.predict(Y)

    def cluster(self):
        self.logger.debug("Starting clustering process")

        self.remove_loops()
        self.build_laplacian()
        self.compute_eigenvalues()

        clusters = self.compute_kmeans()
        self.latest_clusters = clusters

        return self.latest_clusters

    # external connectivitiy
    def get_inter_cluster_edges_count(self, cluster_id: int) -> int:
        # Get edges from a specific cluster to other clusters
        inter_cluster_edges = 0
        cluster_nodes = np.where(self.latest_clusters == cluster_id)[0]
        for node in cluster_nodes:
            # Get neighbors of this node
            neighbors = self.graph[node].indices
            for neighbor in neighbors:
                if self.latest_clusters[neighbor] != cluster_id:
                    inter_cluster_edges += 1
        return inter_cluster_edges

    def get_total_inter_cluster_edges_count(self) -> int:
        # Get total inter-cluster edges (count each edge once)
        inter_cluster_edges = 0
        num_nodes = self.latest_clusters.shape[0]
        for node in range(num_nodes):
            neighbors = self.graph[node].indices
            for neighbor in neighbors:
                # Only count edge once (when node < neighbor)
                if (
                    neighbor > node
                    and self.latest_clusters[node] != self.latest_clusters[neighbor]
                ):
                    inter_cluster_edges += 1
        return inter_cluster_edges

    def get_expansion_ratio(self) -> ClusterAnalysisResult:
        expansion_ratio: dict[str, float] = {}
        unique_clusters = np.unique(self.latest_clusters)
        for cluster_id in unique_clusters:
            inter_edges = self.get_inter_cluster_edges_count(int(cluster_id))
            intra_edges = self.get_intra_cluster_edges_count(int(cluster_id))
            cluster_key = str(int(cluster_id))
            if intra_edges > 0:
                expansion_ratio[cluster_key] = inter_edges / intra_edges
            else:
                expansion_ratio[cluster_key] = float("inf") if inter_edges > 0 else 0.0
        return ClusterAnalysisResult(data=expansion_ratio)

    def get_conductance(self) -> ClusterAnalysisResult:
        # Fraction of total edge volume that points outside the cluster
        conductance: dict[str, float] = {}
        unique_clusters = np.unique(self.latest_clusters)
        for cluster_id in unique_clusters:
            inter_edges = self.get_inter_cluster_edges_count(int(cluster_id))
            intra_edges = self.get_intra_cluster_edges_count(int(cluster_id))
            total_cluster_edges = inter_edges + intra_edges
            cluster_key = str(int(cluster_id))
            if total_cluster_edges > 0:
                conductance[cluster_key] = inter_edges / total_cluster_edges
            else:
                conductance[cluster_key] = 0.0
        return ClusterAnalysisResult(data=conductance)

    @staticmethod
    def calculate_modularity(
        adjacency_matrix: csr_matrix, clusters: np.ndarray
    ) -> float:
        """
        Calculate the modularity Q score for a given clustering.

        Formula: Q = (1/2m) * sum_{i,j} [A_{ij} - (d_i * d_j)/(2m)] * δ(c_i, c_j)

        Where:
        - A_{ij} is the adjacency matrix
        - d_i, d_j are the degrees of nodes i and j
        - m is the total number of edges
        - δ(c_i, c_j) is 1 if nodes i and j are in the same cluster, 0 otherwise

        Args:
            adjacency_matrix: The graph adjacency matrix (sparse)
            clusters: Cluster assignments for each node (n_nodes,)

        Returns:
            The modularity score (float)
        """
        # Calculate degrees
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()

        # Total number of edges (divide by 2 for undirected graph)
        m = adjacency_matrix.sum() / 2.0
        if m == 0:
            return 0.0

        A = (
            adjacency_matrix.toarray()
            if hasattr(adjacency_matrix, "toarray")
            else adjacency_matrix
        )

        cluster_matrix = (clusters[:, np.newaxis] == clusters[np.newaxis, :]).astype(
            float
        )

        degree_outer = np.outer(degrees, degrees)
        B = A - degree_outer / (2.0 * m)

        Q = (1.0 / (2.0 * m)) * np.sum(B * cluster_matrix)

        return float(Q)

    def get_modularity(self) -> float:
        """
        Calculate the modularity Q score for the current clustering.

        Returns:
            The modularity score (float)
        """
        if self.latest_clusters is None:
            self.logger.error("No clusters found. Run cluster() first.")
            return 0.0

        Q = self.calculate_modularity(self.graph, self.latest_clusters)
        self.logger.debug(f"Computed modularity Q = {Q:.6f}")
        return Q

    # internal connectivitiy
    def get_intra_cluster_edges_count(self, cluster_id: int) -> int:
        # Get edges within a specific cluster
        cluster_nodes = np.where(self.latest_clusters == cluster_id)[0]
        intra_cluster_edges = 0
        for node in cluster_nodes:
            neighbors = self.graph[node].indices
            for neighbor in neighbors:
                # Only count edge once (when node < neighbor)
                if neighbor > node and self.latest_clusters[neighbor] == cluster_id:
                    intra_cluster_edges += 1
        return intra_cluster_edges

    def get_total_intra_cluster_edges_count(
        self, normalized: bool = False
    ) -> dict[int, int]:
        # Get edges for all clusters
        cluster_edges_count: dict[int, int] = {}
        unique_clusters = np.unique(self.latest_clusters)
        for cluster_id in unique_clusters:
            cluster_nodes = np.where(self.latest_clusters == cluster_id)[0]
            intra_cluster_edges = 0
            for node in cluster_nodes:
                neighbors = self.graph[node].indices
                for neighbor in neighbors:
                    # Only count edge once (when node < neighbor)
                    if neighbor > node and self.latest_clusters[neighbor] == cluster_id:
                        intra_cluster_edges += 1
            cluster_edges_count[int(cluster_id)] = intra_cluster_edges

        if normalized:
            total_edges = self.graph.nnz // 2  # Divide by 2 since graph is undirected
            if total_edges > 0:
                return {k: v / total_edges for k, v in cluster_edges_count.items()}
            else:
                return cluster_edges_count
        return cluster_edges_count

    def get_triangle_participation_ratio(self) -> dict[int, float]:
        pass
