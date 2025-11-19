from scipy.sparse import csr_matrix, spdiags
import numpy as np


class ClusterMachine:
    def __init__(self, graph: csr_matrix):
        self.graph = graph  # Graph adjacency matrix (Represents the affinity matrix)
        self.laplacian: csr_matrix | None = None

    def remove_loops(self):
        # Remove loops from the graph by subtracting the diagonal of the graph from itself
        self.graph = self.graph - spdiags(
            self.graph.diagonal(), [0], self.graph.shape[0], self.graph.shape[1]
        )

    def build_degree_matrix(self):
        # D is the diagonal matrix who's (i, j) element is the sum of the i-th row of the graph
        D = spdiags(
            self.graph.sum(axis=1), [0], self.graph.shape[0], self.graph.shape[1]
        )
        return D

    def build_laplacian(self):
        # Build the Laplacian matrix L = D^(-1/2) * A * D^(-1/2)
        D = self.build_degree_matrix()
        Dm1f2 = spdiags(1 / np.sqrt(D), [0], self.graph.shape[0], self.graph.shape[1])
        self.laplacian = Dm1f2 * self.graph * Dm1f2

    def cluster(self):
        pass
