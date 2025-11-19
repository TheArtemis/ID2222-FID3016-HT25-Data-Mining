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

        i = self.edges[:, 0] - 1  # Convert 1-based to 0-based indexing
        j = self.edges[:, 1] - 1  # Convert 1-based to 0-based indexing
        n = self.edges.max()  # Max node ID (1-based)
        self.n = n
        self.logger.debug(f"Building graph with {n} nodes and {len(self.edges)} edges")

        matrix = csr_matrix((np.ones(len(self.edges)), (i, j)), shape=(n, n))
        if undirected:
            matrix = matrix + matrix.T

        self.last_build = matrix
        return matrix
