from scipy.stats import bernoulli
from collections.abc import Callable
from collections import defaultdict
import logging
from functools import reduce

logger = logging.getLogger(__name__)


# function needed to parse the file to find the edges
def _get_edge(line: str) -> frozenset[int]:
    return frozenset(
        [int(vertex) for vertex in line.split()]
    )  # frozenset to avoid duplicates and make it "immutable"


class Triest:
    def __init__(self, M: int):
        # M is the size of the memory
        self.M: int = M
        self.t: int = 0  # as indicated by the paper
        self.tau_vertices: defaultdict[int, int] = defaultdict(
            int
        )  # it is a dictionary with int default value
        self.tau: int = 0  # as indicatwed by the paper
        self.S: set[frozenset[int]] = set()

    def _sample_edge(self, t: int) -> bool:
        """This function will determine if the new edge could be added to the graph
        using the reservoir sampling logic"""

        if t <= self.M:
            return True  # Ww still have space in memory
        elif bernoulli.rvs(
            p=self.M / t
        ):  # returns a single bernoulli value(either 0 or 1)
            # remove edge
            return True
        else:
            return False

    @property
    def xi(self) -> float:
        return max(
            1.0,
            self.t
            * (self.t - 1)
            * (self.t - 2)
            / (self.M * (self.M - 1) * (self.M - 2)),
        )

    def _update_counters(
        self, operator: Callable[[int, int], int], edge: frozenset[int]
    ):
        """
        This function updates the counters related to estimating the number of triangles. The update happens
        through the operator and involves the edge and its neighbours.

        :param operator: the lambda used to update the counters
        :param edge: the edge interested in the update
        """
        common_neighbourhood: set[int] = reduce(
            lambda a, b: a & b,
            [
                {
                    node
                    for link in self.S
                    if vertex in link
                    for node in link
                    if node != vertex
                }
                for vertex in edge
            ],
        )

        # I update all the counters by either adding or removing
        for vertex in common_neighbourhood:
            self.tau = operator(self.tau, 1)
            self.tau_vertices[vertex] = operator(self.tau_vertices[vertex], 1)

            for node in edge:
                self.tau_vertices[node] = operator(self.tau_vertices[node], 1)


class TriestBase(Triest):
    """
    This class implements the algorithm Triest base presented in the paper

    The algorithm provides an estimate of the number of triangles in a graph in a streaming environment,
    where the stream represent a series of edges.

    """

    def run(self, file_path: str) -> float:
        # Reset all values
        self.t = 0
        self.tau = 0
        self.tau_vertices.clear()
        self.S.clear()

        logger.info(f"Running the algorithm with M = {self.M}")

        # open the file
        with open(file_path) as f:
            logger.info("Processing the stream directly from the file")
            for line in f:
                edge = _get_edge(line)
                self.t += 1

                if self._sample_edge(self.t):
                    self.S.add(edge)
                    self._update_counters(lambda x, y: x + y, edge)

                if self.t % 1000 == 0:
                    logger.info(
                        f"The current estimate for the number of triangles is {self.xi * self.tau}."
                    )

        return self.xi * self.tau


class TriestImproved(Triest):
    """
    This class implements the improved Triest presented in the paper

    The algorithm provides an estimate of the number of triangles in a graph in a streaming environment,
    where the stream represent a series of edges.

    """

    @property
    def eta(self) -> float:
        return max(1, (self.t - 1) * (self.t - 2) / (self.M * (self.M - 1)))

    def run(self, file_path: str) -> float:
        # Reset all values
        self.t = 0
        self.tau = 0
        self.tau_vertices.clear()
        self.S.clear()

        logger.info(f"Running the algorithm with M = {self.M}")

        # open the file
        with open(file_path) as f:
            logger.info("Processing the stream directly from the file")
            for line in f:
                edge = _get_edge(line)
                self.t += 1

                # new position of update counters
                self._update_counters(lambda x, y: x + y, edge)

                if self._sample_edge(self.t):
                    self.S.add(edge)

                if self.t % 1000 == 0:
                    logger.info(
                        f"The current estimate for the number of triangles is {self.tau}."
                    )

        return self.tau
