import logging
from pathlib import Path
import csv
import sys

from pyspark.sql import SparkSession

from miner.core.similarity.minhashing import Minhashing
from miner.core.similarity.shingling import Shingling

spark = SparkSession.builder.appName("test_books").getOrCreate()

# Reduce Spark's logging verbosity
spark.sparkContext.setLogLevel("WARN")

logger = logging.getLogger(__name__)

# Some story content fields are very large; increase CSV field limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10 * 1024 * 1024)


def test_books():
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    db_books_path = data_dir / "db_books.csv"
    stories_path = data_dir / "stories.csv"

    # Read db_books.csv
    with db_books_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # Build a simple matrix containing the lines of db_books.csv referring to Charles Dickens books.
    dickens_matrix = []
    for r in rows:
        author = r.get("Author", "").strip()
        if author == "Charles Dickens":
            bookno = r.get("bookno", "").strip()
            title = r.get("Title", "").strip()
            language = r.get("Language", "").strip()
            dickens_matrix.append([bookno, title, author, language])

    # Set of Carles Dickens book numbers
    booknos_in_matrix = {row[0] for row in dickens_matrix}

    # Load a filtered version of stories.csv into a dictionary keyed by booknumber
    stories = {}
    with stories_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        # Try to be flexible about header presence
        header = next(reader)
        if [h.lower().strip() for h in header] != ["bookno", "content"]:
            try:
                bookno, content = header
            except Exception:
                bookno, content = None, None
            if bookno and bookno.strip() in booknos_in_matrix:
                stories[bookno.strip()] = content

        for row in reader:
            if not row:
                continue
            # Some rows may contain commas in content; join remaining columns
            bookno = row[0].strip()
            if bookno not in booknos_in_matrix:
                continue
            content = ",".join(row[1:]) if len(row) > 1 else ""
            stories[bookno] = content

    documents = [stories[bookno] for bookno in stories.keys()]

    shingling = Shingling(spark)

    shingles = shingling.shingle_multi(documents)

    # Need to maintain a "per document" logic
    hash_multi = shingling.hash_multi(shingles)

    minhashing = Minhashing(spark, n=100)
    signatures = [minhashing.get_signature(h) for h in hash_multi]

    # Debugging purposes
    print(f"{len(signatures)}")

    # Need to implement something for multi signatures (Guess)
    # The commented part below is just for helping me while writing the test using a trace

    # sh = LSH(spark, n=100, r=r)
    # candidate_pairs = lsh.get_candidate_pairs_spark(signatures)
    # logger.info(f"Candidate pairs for r={r}: {candidate_pairs}")

    # compare_signatures = CompareSignatures(spark)
    # for x, y in candidate_pairs:
    # result = compare_signatures.compare_signatures_spark(
    #    signatures[x], signatures[y]
    # )
    # logger.info(f"Similarity rate = {result:.2f}")


if __name__ == "__main__":
    test_books()
