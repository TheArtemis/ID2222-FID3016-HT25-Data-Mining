from miner.core.similarity.minhashing import Minhashing
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Minhashing").getOrCreate()


def test_minhashing():
    minhashing = Minhashing(spark, n=100)
    p = minhashing.get_row_permutation(0)
    print(p)
    assert len(p) == 100

    p1 = minhashing.get_row_permutation(1)
    print(p1)
    assert len(p1) == 100
    assert p1 != p


if __name__ == "__main__":
    test_minhashing()
