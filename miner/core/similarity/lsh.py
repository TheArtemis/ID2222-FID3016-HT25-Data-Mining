import logging
import math
from pyspark.sql import SparkSession
from pydantic import BaseModel
import mmh3


class Band(BaseModel):
    rows: list[list[int]]

    def __len__(self):
        return len(self.rows)

    def get_n_columns(self):
        return len(self.rows[0])

    def get_column(self, i: int) -> list[int]:
        return [row[i] for row in self.rows]

    def hash_column(self, i: int) -> int:
        return mmh3.hash(",".join(map(str, self.get_column(i))))


class Bucket(BaseModel):
    # key: hash value, value: list of the column indices (documents) that have this hash
    hash_dict: dict[int, list[int]]

    def add_hash(self, i: int, value: int):
        if value not in self.hash_dict:
            self.hash_dict[value] = [i]
        else:
            self.hash_dict[value].append(i)

    def get_candidates(self):
        return [doc_idxs for doc_idxs in self.hash_dict.values() if len(doc_idxs) > 1]

    def __repr__(self) -> str:
        counts_repr = ", ".join(
            f"{hash_value}:{len(doc_indices)}"
            for hash_value, doc_indices in sorted(self.hash_dict.items())
        )
        return f"Bucket({{{counts_repr}}})"


class LSH:
    def __init__(self, spark: SparkSession, n: int, r: int):
        self.logger = logging.getLogger(__name__)
        self.spark = spark

        self.n = n  # Signature length
        self.b = math.ceil(n / r)  # Number of bands
        self.r = r  # Number of rows in each band

    def get_candidate_pairs(self, signatures: list[list[int]]):
        buckets = self.create_buckets(signatures)
        candidate_pairs: set[(int, int)] = set()

        for bucket in buckets:
            docs = bucket.get_candidates()
            for doc_idxs in docs:
                for i in doc_idxs:
                    for j in doc_idxs:
                        if i < j:
                            candidate_pairs.add((i, j))
        return candidate_pairs

    def create_buckets(self, signatures: list[list[int]]):
        buckets: list[Bucket] = []
        for band in self.get_bands(signatures):
            bucket = Bucket(hash_dict={})
            for i in range(band.get_n_columns()):
                hashed_col = band.hash_column(i)
                bucket.add_hash(i, hashed_col)
            buckets.append(bucket)

        return buckets

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
