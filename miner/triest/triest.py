from scipy.stats import bernoulli

class Triest:
    "needed for Triest triangle estimation method"
    def __init__(self, file:str, M:int):
        # M is the size of the memory
        # str the file to be read
        self.file: str = file
        self.M: int = M
        self.t: int = 0
        self.tau: int = 0

    def _sample_edge(self, t:int) -> bool:
        "This function will determine if the new edge could be added to the graph"
        "using the reservoir sampling logic"

        if t <= self.M:
            return True #Ww still have space in memory
        elif bernoulli.rvs(p = self.M / t): #returns a single bernoulli value(either 0 or 1)
            # remove edge
            return True
        else:
            return False
        
