from pyspark.sql import SparkSession
from miner.core.similarity.compare_signatures import CompareSignatures

spark = SparkSession.builder.appName("Test Compare Signatures").getOrCreate()


def test_compare_signatures():
    compare_signatures = CompareSignatures(spark)
    sign1 = [1, 0, 1, 2, 3, 4]
    sign2 = [1, 2, 2, 2, 4, 4]
    similarity = compare_signatures.compare_signatures(sign1, sign2)
    print(f"Similarity: {similarity:.2f}")


if __name__ == "__main__":
    test_compare_signatures()
