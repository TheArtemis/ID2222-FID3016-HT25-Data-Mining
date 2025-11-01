from enum import Enum
import re
from pyspark.sql import SparkSession
import mmh3


class ShingleSize(Enum):
    BIG = 9  # For big documents
    SMALL = 5  # For small documents such as emails and short texts


class Shingling:
    def __init__(self, spark: SparkSession, k: int = ShingleSize.SMALL.value):
        self.spark = spark
        self.k = k

    def shingle(self, document: str) -> list[str]:
        document = self._preprocess_document(document)
        shingles = [document[i : i + self.k] for i in range(len(document) - self.k + 1)]
        return shingles

    def hash_shingles(self, shingles: list[str], duplicates: bool = False) -> list[int]:
        rdd = self.spark.sparkContext.parallelize(shingles)

        hashed_shingles = rdd.map(Shingling.hash_shingle)
        result = hashed_shingles.collect()
        if duplicates:
            return result
        else:
            return list[int](dict.fromkeys(result))

    def _preprocess_document(self, document: str) -> str:
        # Replace consecutive whitespaces with a single space
        document = re.sub(r"\s+", " ", document)

        # Normalize to lowercase
        document = document.lower()

        # Remove punctuation
        document = re.sub(r"[^\w\s]", "", document)

        return document

    @staticmethod
    def hash_shingle(shingle: str) -> int:
        # We use mmh3 to hash the shingle and then we mask the result to 32 bits
        # Set the seed for consistent hashing across runs
        return mmh3.hash(shingle, seed=0) & 0xFFFFFFFF

    # TODO Handle multiple documents
