from scipy.stats import bernoulli
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)
Edge = frozenset[int]


# function needed to parse the file to find the edges
def _get_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("Self-loop")
    return frozenset((int(u), int(v)))


class TriestBase:
    """
    This class implements the algorithm Triest base presented in the paper

    The algorithm provides an estimate of the number of triangles in a graph in a streaming environment,
    where the stream represent a series of edges.

    """

    def __init__(self, M: int):
        # M is the size of the memory
        self.M: int = M
        self.t: int = 0  # as indicated by the paper
        self.neighbors: defaultdict[int, set[int]] = defaultdict(set)
        self.tau_vertices: defaultdict[int, float] = defaultdict(
            float
        )  # it is a dictionary with int default value
        self.tau: float = 0.0  # as indicated by the paper
        self.S: set[Edge] = set()

    def reset(self):
        self.t = 0
        self.S: set[Edge] = set()
        self.neighbors: defaultdict[int, set[int]] = defaultdict(set)
        self.tau = 0.0
        self.tau_vertices: defaultdict[int, float] = defaultdict(float)

    @property
    def xi(self) -> float:
        return max(
            1.0,
            self.t
            * (self.t - 1)
            * (self.t - 2)
            / (self.M * (self.M - 1) * (self.M - 2)),
        )

    def _add_remove_edge_to_sample(self, e: Edge, b: bool):
        u, v = tuple(e)
        if b:
            self.S.add(e)
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
        else:
            self.S.remove(e)
            self.neighbors[u].remove(v)
            self.neighbors[v].remove(u)

    def _common_neighbours(self, e: Edge) -> set[int]:
        u, v = tuple(e)
        return self.neighbors[u] & self.neighbors[v]

    def _sample_edge(self, e: Edge) -> tuple[bool, Edge | None]:
        if self.t <= self.M:
            return True, None
        accept_prob = self.M / float(self.t)
        # Bernoulli trial to decide whether to keep or discard
        if bernoulli.rvs(p=accept_prob):
            if not self.S:
                return True, None
            evict = random.choice(tuple(self.S))
            return True, evict
        return False, None

    def _update_counters(self, e: Edge, inc: float):
        if len(self.S) == 0:
            return

        u, v = tuple(e)
        common = self._common_neighbours(e)
        if not common:
            return

        for c in common:
            self.tau += inc
            self.tau_vertices[u] += inc
            self.tau_vertices[v] += inc

    def estimate_global(self) -> float:
        return self.xi * self.tau

    def estimate_local(self, u: int) -> float:
        return self.xi * self.tau_vertices.get(u, 0.0)

    def process_insertion(self, u: int, v: int):
        e = _get_edge(u, v)
        self.t += 1

        accepted, evicted = self._sample_edge(e)
        if accepted:
            if evicted is not None:
                self._update_counters(evicted, -1.0)
                self._add_remove_edge_to_sample(evicted, False)

            self._add_remove_edge_to_sample(e, True)
            self._update_counters(e, +1.0)

    def run(self, file_path: str) -> float:
        self.reset()
        logger.info(f"Running the algorithm with M = {self.M}")

        # open the file and read edges
        edges: list[tuple[int, int]] = []
        with open(file_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                edges.append((u, v))
        for u, v in edges:
            self.process_insertion(u, v)
        return self.estimate_global


class TriestImproved(TriestBase):
    """
    This class implements the improved Triest presented in the paper

    The algorithm provides an estimate of the number of triangles in a graph in a streaming environment,
    where the stream represent a series of edges.

    """

    @property
    def eta(self) -> float:
        return max(1.0, (self.t - 1) * (self.t - 2) / (self.M * (self.M - 1)))

    def estimate_global(self) -> float:
        return float(self.tau)

    def process_insertion(self, u: int, v: int):
        e = _get_edge(u, v)
        self.t += 1

        self._update_counters(e, self.eta)
        accepted, evicted = self._sample_edge(e)
        if accepted:
            if evicted is not None:
                self._add_remove_edge_to_sample(evicted, False)
            self._add_remove_edge_to_sample(e, True)

    def run(self, file_path: str) -> float:
        # Reset all values
        self.reset()

        logger.info(f"Running the algorithm with M = {self.M}")

        # open the file
        edges: list[tuple[int, int]] = []
        with open(file_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                edges.append((u, v))

        for u, v in edges:
            self.process_insertion(u, v)
        return self.estimate_global()
