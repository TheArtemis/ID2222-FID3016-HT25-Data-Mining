import logging
from pyspark.sql import SparkSession
from miner.core.similarity.lsh import LSH


s1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
s2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
s3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
s4 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
s5 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

spark = SparkSession.builder.appName("test_lsh").getOrCreate()

logger = logging.getLogger(__name__)


def test_bands():
    lsh = LSH(spark=spark, n=11, r=2)
    bands = lsh.get_bands(signatures=[s1, s2, s3, s4, s5])

    logger.info(bands)


if __name__ == "__main__":
    test_bands()
