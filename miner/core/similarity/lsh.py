import logging
import math
from pyspark.sql import SparkSession
from pydantic import BaseModel
import mmh3


class Band(BaseModel):
    rows: list[list[int]]

    def __len__(self):
        return len(self.rows)

    def get_column(self, i: int) -> list[int]:
        return [row[i] for row in self.rows]

    def hash_column(self, i: int) -> int:
        return mmh3.hash(",".join(map(str, self.get_column(i))))


class Bucket(BaseModel):
    pass


class LSH:
    def __init__(self, spark: SparkSession, n: int, r: int):
        self.logger = logging.getLogger(__name__)
        self.spark = spark

        self.n = n  # Signature length
        self.b = math.ceil(n / r)  # Number of bands
        self.r = r  # Number of rows in each band

    def get_candidate_pairs(self, signatures: list[list[int]]):
        pass

    def get_bands(self, signatures: list[list[int]]):
        bands = []
        # Create bands by chunking all n rows into groups of size r
        for start in range(0, self.n, self.r):
            end = min(start + self.r, self.n)
            rows = []
            for row_idx in range(start, end):
                row = [
                    signatures[col_idx][row_idx] for col_idx in range(len(signatures))
                ]
                rows.append(row)
            bands.append(Band(rows=rows))
        return bands
