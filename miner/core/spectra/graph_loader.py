from scipy.sparse import csr_matrix
import logging
import numpy as np


class GraphLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.edges = None
        self.n = None
        self.last_build: csr_matrix | None = None
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        with open(self.file_path) as f:
            self.edges = np.loadtxt(f, delimiter=",", dtype=int)

    def build(self, undirected: bool = True) -> csr_matrix:
        if not self.edges:
            self.load_data()

        i = self.edges[:, 0] - 1
        j = self.edges[:, 1] - 1
        n = self.edges.max()
        self.n = n
        self.logger.debug(f"Building graph with {n} nodes and {len(self.edges)} edges")

        matrix = csr_matrix((np.ones(len(self.edges)), (i, j)), shape=(n, n))

        if undirected:
            # Check if matrix is already symmetric by comparing non-zero patterns
            # For sparse matrices, we check if (matrix != matrix.T) has any non-zeros
            diff = matrix - matrix.T
            is_symmetric = diff.nnz == 0

            if not is_symmetric:
                # Make symmetric by taking maximum (for binary graphs, this is equivalent to OR)
                # This ensures we don't double weights if edges appear in both directions
                matrix = matrix.maximum(matrix.T)
                self.logger.debug("Made graph symmetric (undirected)")
            else:
                self.logger.debug("Graph is already symmetric")

        self.last_build = matrix
        return matrix
