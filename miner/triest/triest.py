from scipy.stats import bernoulli
import random 
from typing import FrozenSet, Set
import logging

logger = logging.getLogger(__name__)

class Triest:
    "needed for Triest triangle estimation method"
    def __init__(self, file:str, M:int):
        # M is the size of the memory
        # str the file to be read
        self.file: str = file
        self.M: int = M
        self.t: int = 0 #as indicated by the paper
        self.tau: int = 0 #as indicatwed by the paper
        self.S = set[FrozenSet[int]] = set()

    def _sample_edge(self, t:int) -> bool:
        "This function will determine if the new edge could be added to the graph"
        "using the reservoir sampling logic"

        if t <= self.M:
            return True #Ww still have space in memory
        elif bernoulli.rvs(p = self.M / t): #returns a single bernoulli value(either 0 or 1)
            edge_to_remove: FrozenSet[int] = random.choice(self.S)
            self.S.remove(edge_to_remove)
            self._update_counters(lambda x, y: x - y, edge_to_remove)
            return True
        else:
            return False
        
    def xi(self) -> float:
        return max(1.0,
                   self.t*(self.t-1)*(self.t-2)/
                   (self.M*(self.M-1)*(self.M-2))
                   )
    
    def _update_counters(self, operator: Callable[[int,int], int], edge: FrozenSet[int]):
        """
        This function updates the counters related to estimating the number of triangles. The update happens 
        through the operator and involves the edge and its neighbours.

        :param operator: the lambda used to update the counters
        :param edge: the edge interested in the update
        """
        # build neighbour sets per vertex
        neighbour_sets = [
            {node for link in self.S if vertex in link for node in link if node != vertex}
            for vertex in edge
        ]

        # handle empty edge or no sets
        common_neighbourhood = set.intersection(*neighbour_sets) if neighbour_sets else set()

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

    def run(self) -> float:
        logger.info(f"Running the algorithm with M = {self.M}")







        
