from pathlib import Path
from miner.core.similarity.shingling import Shingling
from miner.core.similarity.minhashing import Minhashing
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Minhashing").getOrCreate()


DATA_FOLDER_PATH = Path(__file__).parent.parent / "data"
EMAIL_FOLDER_PATH = DATA_FOLDER_PATH / "email"
EMAIL_1 = EMAIL_FOLDER_PATH / "email1.txt"
EMAIL_2 = EMAIL_FOLDER_PATH / "email2.txt"
EMAIL_3 = EMAIL_FOLDER_PATH / "email3.txt"


def test_minhashing():
    minhashing = Minhashing(spark, n=100)

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

    print("Getting signatures...")

    signature1 = minhashing.get_signature(h1)
    signature2 = minhashing.get_signature(h2)
    signature3 = minhashing.get_signature(h3)

    print(signature1)
    print(signature2)
    print(signature3)


if __name__ == "__main__":
    test_minhashing()
