import logging
from pathlib import Path

from pyspark.sql import SparkSession

from miner.core.similarity import LSH
from miner.core.similarity.minhashing import Minhashing
from miner.core.similarity.shingling import Shingling


DATA_FOLDER_PATH = Path(__file__).parent.parent.parent / "data"
EMAIL_FOLDER_PATH = DATA_FOLDER_PATH / "email"
EMAIL_1 = EMAIL_FOLDER_PATH / "email1.txt"
EMAIL_2 = EMAIL_FOLDER_PATH / "email2.txt"
EMAIL_3 = EMAIL_FOLDER_PATH / "email3.txt"
EMAIL_1_COPY = EMAIL_FOLDER_PATH / "email1_copy.txt"

spark = SparkSession.builder.appName("test_lsh").getOrCreate()

logger = logging.getLogger(__name__)


def test_lsh(r: int = 10):
    e1 = open(EMAIL_1).read()
    e2 = open(EMAIL_2).read()
    e3 = open(EMAIL_3).read()

    shingling = Shingling(spark)
    shingles1 = shingling.shingle(e1)
    shingles2 = shingling.shingle(e2)
    shingles3 = shingling.shingle(e3)

    h1 = shingling.hash_shingles(shingles1)
    h2 = shingling.hash_shingles(shingles2)
    h3 = shingling.hash_shingles(shingles3)

    minhashing = Minhashing(spark, n=100)
    signature1 = minhashing.get_signature(h1)
    signature2 = minhashing.get_signature(h2)
    signature3 = minhashing.get_signature(h3)

    lsh = LSH(spark, n=100, r=r)
    candidate_pairs = lsh.get_candidate_pairs(
        signatures=[signature1, signature2, signature3]
    )
    logger.info(f"Candidate pairs for r={r}: {candidate_pairs}")


def test_lsh_with_copy():
    e1 = open(EMAIL_1).read()
    e1_copy = open(EMAIL_1_COPY).read()

    shingling = Shingling(spark)
    shingles1 = shingling.shingle(e1)
    shingles1_copy = shingling.shingle(e1_copy)

    h1 = shingling.hash_shingles(shingles1)
    h1_copy = shingling.hash_shingles(shingles1_copy)

    minhashing = Minhashing(spark, n=100)
    signature1 = minhashing.get_signature(h1)
    signature1_copy = minhashing.get_signature(h1_copy)

    lsh = LSH(spark, n=100, r=10)
    bands = lsh.get_bands(signatures=[signature1, signature1_copy])
    buckets = lsh.create_buckets(signatures=[signature1, signature1_copy])
    logger.info(bands)
    logger.info(buckets)
    candidate_pairs = lsh.get_candidate_pairs(signatures=[signature1, signature1_copy])
    logger.info(candidate_pairs)


def test_lsh_diff_r():
    for r in range(1, 30, 2):
        test_lsh(r)


if __name__ == "__main__":
    test_lsh_diff_r()
