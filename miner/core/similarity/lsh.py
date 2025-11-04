import logging
import math
from pyspark.sql import SparkSession
from pydantic import BaseModel
import mmh3
from collections.abc import Iterable


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

    @staticmethod
    def _generate_pairs(doc_idxs: list[int]) -> list[(int, int)]:
        pairs = []
        if not doc_idxs or len(doc_idxs) < 2:
            return pairs
        for i in doc_idxs:
            for j in doc_idxs:
                if i < j:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def _col_to_band_entries(
        col_item: tuple[int, list[int]], b: int, r: int, n: int, mmh3_hash
    ) -> list[tuple[int, tuple[int, int]]]:
        col_idx, sig = col_item
        entries = []
        for band_idx in range(b):
            start = band_idx * r
            end = min(start + r, n)
            band_values = sig[start:end]
            hashed = mmh3_hash(",".join(map(str, band_values)))
            entries.append((band_idx, (hashed, col_idx)))
        return entries

    @staticmethod
    def _assemble_band_rows(
        iterable: Iterable[tuple[int, list[int]]],
    ) -> list[list[int]]:
        rows_map: dict[int, list[int]] = {}
        for row_within, row_values in iterable:
            rows_map[int(row_within)] = row_values
        return [rows_map[i] for i in sorted(rows_map.keys())]

    def get_candidate_pairs(self, signatures: list[list[int]]) -> set[(int, int)]:
        buckets = self.create_buckets(signatures)
        candidate_pairs: set[(int, int)] = set()

        for bucket in buckets:
            docs = bucket.get_candidates()
            for doc_idxs in docs:
                pairs = LSH._generate_pairs(doc_idxs)
                candidate_pairs.update(pairs)
        return candidate_pairs

    def get_candidate_pairs_spark(self, signatures: list[list[int]]) -> set[(int, int)]:
        # Create buckets
        buckets = self.create_buckets(signatures)

        # Create a list to parallelize
        docs = []
        for bucket in buckets:
            docs.extend(bucket.get_candidates())
        rdd = self.spark.sparkContext.parallelize(docs)

        # Generate all unique candidate pairs
        pairs_rdd = rdd.flatMap(LSH._generate_pairs)

        # Eliminate duplicates
        pairs_rdd = pairs_rdd.distinct()

        # Collect
        candidate_pairs: set[(int, int)] = set(pairs_rdd.collect())

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

    def create_buckets_spark(self, signatures: list[list[int]]):
        b = int(self.b)
        r = int(self.r)
        n = int(self.n)
        mmh3_hash = mmh3.hash

        cols = list(enumerate(signatures))
        rdd = self.spark.sparkContext.parallelize(cols)

        pairs = rdd.flatMap(
            lambda item: LSH._col_to_band_entries(item, b, r, n, mmh3_hash)
        )  # (band_idx, (hashed, col_idx))

        def build_hash_dict(values: Iterable[tuple]):
            d: dict[int, list[int]] = {}
            for hashed, col_idx in values:
                d.setdefault(hashed, []).append(col_idx)
            return d

        bands_rdd = pairs.groupByKey().mapValues(build_hash_dict)
        band_dict = bands_rdd.collectAsMap()

        buckets: list[Bucket] = []
        for band_idx in range(b):
            hash_dict = band_dict.get(band_idx, {})
            buckets.append(Bucket(hash_dict=hash_dict))
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

    def get_bands_spark(self, signatures: list[list[int]]):
        b = int(self.b)
        r = int(self.r)
        n = int(self.n)

        # Broadcast the signatures to avoid copying for each task
        signatures_b = self.spark.sparkContext.broadcast(signatures)

        def row_to_band_tuple_local(row_idx: int):
            band_idx = row_idx // r
            row_within = row_idx % r
            sigs = signatures_b.value
            row_values = [sigs[col_idx][row_idx] for col_idx in range(len(signatures))]
            return (band_idx, (row_within, row_values))

        rows_rdd = self.spark.sparkContext.parallelize(range(n)).map(
            row_to_band_tuple_local
        )
        bands_rdd = rows_rdd.groupByKey().mapValues(LSH._assemble_band_rows)
        band_dict = bands_rdd.collectAsMap()

        bands = [Band(rows=band_dict.get(bi, [])) for bi in range(b)]
        signatures_b.unpersist()
        return bands
