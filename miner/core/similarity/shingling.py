from enum import Enum
import logging
import re
from pyspark.sql import SparkSession
import mmh3

from miner.settings import SEED, SHINGLE_PYSPARK_THRESHOLD


class ShingleSize(Enum):
    BIG = 9  # For big documents
    SMALL = 5  # For small documents such as emails and short texts


class Shingling:
    def __init__(self, spark: SparkSession, k: int = ShingleSize.SMALL.value):
        self.logger = logging.getLogger(__name__)
        self.spark = spark
        self.k = k

    def shingle(self, document: str) -> list[str]:
        document = self._preprocess_document(document)
        shingles = [document[i : i + self.k] for i in range(len(document) - self.k + 1)]
        return shingles

    def hash_shingles(self, shingles: list[str], duplicates: bool = False) -> list[int]:
        result = []
        if len(shingles) > SHINGLE_PYSPARK_THRESHOLD:
            self.logger.info(f"Using pyspark to hash {len(shingles)} shingles")
            result = self._hash_shingles_pyspark(shingles)
        else:
            result = [Shingling.hash_shingle(shingle) for shingle in shingles]

        if duplicates:
            return result
        else:
            return list(dict.fromkeys(result))

    def _hash_shingles_pyspark(self, shingles: list[str]) -> list[int]:
        rdd = self.spark.sparkContext.parallelize(shingles)
        hashed_shingles = rdd.map(Shingling.hash_shingle)
        return hashed_shingles.collect()

    def _preprocess_document(self, document: str, keep_spaces: bool = True) -> str:
        if keep_spaces:
            # Replace consecutive whitespaces with a single space
            document = re.sub(r"\s+", " ", document)
        else:
            # Remove all whitespaces
            document = re.sub(r"\s", "", document)

        # Normalize to lowercase
        document = document.lower()

        # Remove punctuation
        document = re.sub(r"[^\w\s]", "", document)

        return document

    @staticmethod
    def hash_shingle(shingle: str, seed: int = SEED) -> int:
        # We use mmh3 to hash the shingle and then we mask the result to 32 bits
        # Set the seed for consistent hashing across runs
        return mmh3.hash(shingle, seed=seed) & 0xFFFFFFFF

    # Shingles each document
    def shingle_multi(self, document: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for doc in document:
            result.append(self.shingle(doc))
        return result

    def hash_multi(self, shingles=list[list[str]]) -> list[list[int]]:
        result: list[list[str]] = []

        for shingle in shingles:
            result.append(self.hash_shingles(shingle))

        return result

    def hash_multi_spark(self, shingles=list[list[str]]) -> list[list[int]]:
        result: list[list[str]] = []
        for shingle in shingles:
            result.append(self._hash_shingles_pyspark(shingle))

        return result
