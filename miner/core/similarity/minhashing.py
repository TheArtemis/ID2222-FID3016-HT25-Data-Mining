"""
A class MinHashing that builds a minHash signature
(in the form of a vector or a set)
of a given length n from a given set of integers
(a set of hashed shingles).
"""

from collections.abc import Callable
import logging
import random
from pyspark.sql import SparkSession

from miner.settings import BIG_PRIME, SEED


class Minhashing:
    def __init__(self, spark: SparkSession, n: int, seed: int = SEED):
        self.logger = logging.getLogger(__name__)
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

    def get_signature(self, shingles: list[int]) -> list[int]:
        signature = []

        if len(shingles) > 0:
            self.logger.info(
                f"Using pyspark to compute the signature for {len(shingles)} shingles"
            )
            rdd = self.spark.sparkContext.parallelize(shingles)
            for hash_fn in self.hash_functions:
                min_hash = rdd.map(hash_fn).min()
                signature.append(min_hash)
        else:              
            # This part might easily be taken away since it is basically never used
            for hash_fn in self.hash_functions:
                min_hash = min(shingles, key=hash_fn)
                signature.append(min_hash)

        return signature
