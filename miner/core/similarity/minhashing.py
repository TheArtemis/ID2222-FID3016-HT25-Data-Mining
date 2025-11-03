from collections.abc import Callable
import logging
import random
from pyspark.sql import SparkSession

from miner.settings import BIG_PRIME, SEED, SHINGLE_PYSPARK_THRESHOLD


class Minhashing:
    def __init__(self, spark: SparkSession, n: int, seed: int = SEED):
        self.logger = logging.getLogger(__name__)
        self.spark = spark
        self.n = n  # Signature length
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

    # TODO: On the book there is a way to speed up this, only computing the min hash for the first m shingles or something like this
    def get_signature(self, shingles: list[int]) -> list[int]:
        signature = []

        if len(shingles) > SHINGLE_PYSPARK_THRESHOLD:
            self.logger.info(
                f"Using pyspark to compute the signature for {len(shingles)} shingles"
            )
            rdd = self.spark.sparkContext.parallelize(shingles)
            for hash_fn in self.hash_functions:
                min_hash = rdd.map(hash_fn).min()
                signature.append(min_hash)
        else:
            for hash_fn in self.hash_functions:
                min_hash = min(shingles, key=hash_fn)
                signature.append(min_hash)

        return signature
