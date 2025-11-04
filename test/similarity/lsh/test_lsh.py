import logging
from pathlib import Path

from pyspark.sql import SparkSession

from miner.core.similarity import LSH
from miner.core.similarity.minhashing import Minhashing
from miner.core.similarity.shingling import Shingling
from miner.core.similarity.compare_signatures import CompareSignatures


DATA_FOLDER_PATH = Path(__file__).parent.parent.parent / "data"
EMAIL_FOLDER_PATH = DATA_FOLDER_PATH / "email"
EMAIL_1 = EMAIL_FOLDER_PATH / "email1.txt"
EMAIL_2 = EMAIL_FOLDER_PATH / "email2.txt"
EMAIL_3 = EMAIL_FOLDER_PATH / "email3.txt"
EMAIL_1_COPY = EMAIL_FOLDER_PATH / "email1_copy.txt"

spark = SparkSession.builder.appName("test_lsh").getOrCreate()

# Reduce Spark's logging verbosity
spark.sparkContext.setLogLevel("WARN")

logger = logging.getLogger(__name__)


def test_lsh(r: int = 10):
    emails = [p.read_text() for p in (EMAIL_1, EMAIL_2, EMAIL_3)]

    shingling = Shingling(spark)

    shingles = [shingling.shingle(s) for s in emails]

    hash = [shingling.hash_shingles(s) for s in shingles]

    minhashing = Minhashing(spark, n=100)
    signatures = [minhashing.get_signature(h) for h in hash]

    lsh = LSH(spark, n=100, r=r)
    candidate_pairs = lsh.get_candidate_pairs(signatures)
    logger.info(f"Candidate pairs for r={r}: {candidate_pairs}")

    compare_signatures = CompareSignatures(spark)
    for x, y in candidate_pairs:
        result = compare_signatures.compare_signatures(signatures[x], signatures[y])
        logger.info(f"Similarity rate = {result:.2f}")


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
    logger.info(f"Bands: {bands}")
    logger.info(f"Buckets: {buckets}")
    candidate_pairs_spark = lsh.get_candidate_pairs_spark(
        signatures=[signature1, signature1_copy]
    )

    candidate_pairs = lsh.get_candidate_pairs(signatures=[signature1, signature1_copy])
    assert candidate_pairs == candidate_pairs_spark
    logger.info(f"Candidate pairs: {candidate_pairs}")


def test_lsh_diff_r():
    for r in range(1, 30, 2):
        test_lsh(r)


if __name__ == "__main__":
    test_lsh_with_copy()
