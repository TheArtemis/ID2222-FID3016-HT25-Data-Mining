from pathlib import Path
from pyspark.sql import SparkSession
from miner.core.similarity.shingling import Shingling

DATA_FOLDER_PATH = Path(__file__).parent.parent.parent / "data"

spark = SparkSession.builder.appName("Test Shingling").getOrCreate()


def test_shingling():
    shingling = Shingling(spark)
    document = open(DATA_FOLDER_PATH / "laws_of_robotics.txt").read()
    shingles = shingling.shingle(document)
    hashed_shingles = shingling.hash_shingles(shingles)
    print("Shingles:")
    print(shingles)
    print("-" * 100)
    print("Hashed shingles:")
    print(hashed_shingles)


if __name__ == "__main__":
    test_shingling()
