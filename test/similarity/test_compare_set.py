from pathlib import Path

from pyspark.sql import SparkSession
from miner.core.similarity.compare_sets import CompareSets
from miner.core.similarity.shingling import Shingling

DATA_FOLDER_PATH = Path(__file__).parent.parent / "data"
EMAIL_FOLDER_PATH = DATA_FOLDER_PATH / "email"
EMAIL_1 = EMAIL_FOLDER_PATH / "email1.txt"
EMAIL_2 = EMAIL_FOLDER_PATH / "email2.txt"
EMAIL_3 = EMAIL_FOLDER_PATH / "email3.txt"

spark = SparkSession.builder.appName("Test Compare Sets").getOrCreate()


def test_compare_set():
    compare_set = CompareSets()
    set1 = [1, 2, 3, 4, 5]
    set2 = [4, 5, 6, 7, 8]
    similarity = compare_set.jaccard_similarity(set1, set2)
    print(f"Similarity: {similarity:.2f}")


def test_compare_document():
    shingling = Shingling(spark)

    e1 = open(EMAIL_1).read()
    e2 = open(EMAIL_2).read()
    e3 = open(EMAIL_3).read()

    shingles1 = shingling.shingle(e1)
    shingles2 = shingling.shingle(e2)
    shingles3 = shingling.shingle(e3)

    h1 = shingling.hash_shingles(shingles1)
    h2 = shingling.hash_shingles(shingles2)
    h3 = shingling.hash_shingles(shingles3)

    compare_set = CompareSets()
    h1_h2_similarity = compare_set.jaccard_similarity(h1, h2)
    h1_h3_similarity = compare_set.jaccard_similarity(h1, h3)
    h2_h3_similarity = compare_set.jaccard_similarity(h2, h3)

    print(f"Similarity between e1 and e2: {h1_h2_similarity:.2f}")
    print(f"Similarity between e1 and e3: {h1_h3_similarity:.2f}")
    print(f"Similarity between e2 and e3: {h2_h3_similarity:.2f}")


if __name__ == "__main__":
    test_compare_set()
    test_compare_document()
