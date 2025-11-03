from pathlib import Path
from pyspark.sql import SparkSession
from miner.core.similarity.shingling import Shingling


DATA_FOLDER_PATH = Path(__file__).parent.parent.parent / "data"

spark = SparkSession.builder.appName("Test Shingling Multifile").getOrCreate()

def test_shingling_multifile():
    shingling = Shingling(spark)

    files = ["email1.txt", "email2.txt", "email3.txt"]
    document = [open(DATA_FOLDER_PATH / "email" / f).read() for f in files]

    shingles_multi = shingling.shingle_multi(document)

    hashed_shingles = shingling.hash_multi(shingles_multi)
    for idx, hashed in enumerate(hashed_shingles, start=1):
        print(f"Document {idx} hashed shingles:")
        print(hashed)
        print("-" * 100)


if __name__ == "__main__":
    test_shingling_multifile()