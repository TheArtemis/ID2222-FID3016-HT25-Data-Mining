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

    def build(self, undirected: bool = False) -> csr_matrix:
        if not self.edges:
            self.load_data()

        i = self.edges[:, 0] - 1
        j = self.edges[:, 1] - 1
        n = self.edges.max()
        self.n = n
        self.logger.debug(f"Building graph with {n} nodes and {len(self.edges)} edges")

        matrix = csr_matrix((np.ones(len(self.edges)), (i, j)), shape=(n, n))
        if undirected:
            matrix = matrix + matrix.T

        self.last_build = matrix
        return matrix
