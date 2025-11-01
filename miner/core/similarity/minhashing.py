"""
A class MinHashing that builds a minHash signature
(in the form of a vector or a set)
of a given length n from a given set of integers
(a set of hashed shingles).
"""

from collections.abc import Callable
import random
from pyspark.sql import SparkSession

from miner.settings import BIG_PRIME, SEED


class Minhashing:
    def __init__(self, spark: SparkSession, n: int, seed: int = SEED):
        self.spark = spark
        self.n = n
        self.seed = seed
        self.hash_functions: list[Callable] = self.generate_hash_functions()

    def generate_hash_functions(self) -> list[Callable]:
        # Creates n hash functions of the form f(x) = (ax + b) % p
        random.seed(self.seed)
        hash_functions = []
        for _ in range(self.n):
            a = random.randint(1, BIG_PRIME - 1)
            b = random.randint(0, BIG_PRIME - 1)
            hash_functions.append(lambda x, a=a, b=b: (a * x + b) % BIG_PRIME)

        return hash_functions

    def get_signature(self, i: int) -> list[int]:
        # Return a list of indices sorted by the hash function hash_fn
        return sorted(range(self.n), key=lambda x: self.hash_functions[i](x))
