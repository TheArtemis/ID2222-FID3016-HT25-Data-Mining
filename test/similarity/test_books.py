import logging
from pathlib import Path
import time
import csv
import sys

from pyspark.sql import SparkSession

from miner.core.similarity.lsh import LSH
from miner.core.similarity.minhashing import Minhashing
from miner.core.similarity.shingling import Shingling
from miner.core.similarity.compare_signatures import CompareSignatures

spark = SparkSession.builder.appName("test_books").getOrCreate()

# Reduce Spark's logging verbosity
spark.sparkContext.setLogLevel("WARN")

logger = logging.getLogger(__name__)

# Some story content fields are very large; increase CSV field limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10 * 1024 * 1024)


def test_books(rows_per_band: int = 4):
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

    complete_time_start = time.time()

    shingling = Shingling(spark)

    shingles = shingling.shingle_multi(documents)

    # Need to maintain a "per document" logic
    hash_multi = shingling.hash_multi_spark(shingles)

    shingling_time = time.time()
    logger.info(
        f"Time elapsed for shingling: {shingling_time - complete_time_start:.3f} seconds"
    )

    minhashing_start_time = time.time()

    minhashing = Minhashing(spark, n=100)
    signatures = [minhashing.get_signature(h) for h in hash_multi]

    minhashing_end_time = time.time()
    logger.info(
        f"Time elapsed for minhashing: {minhashing_end_time - minhashing_start_time:.3f} seconds"
    )

    lsh_start_time = time.time()

    lsh = LSH(spark, n=100, r=rows_per_band)
    candidate_pairs = lsh.get_candidate_pairs_spark(signatures)
    logger.info(f"Candidate pairs for r={rows_per_band}: {candidate_pairs}")

    lsh_end_time = time.time()
    logger.info(f"Time elapsed for LSH: {lsh_end_time - lsh_start_time:.3f} seconds")

    print(f"{candidate_pairs}")

    compare_sign_start_time = time.time()

    compare_signatures = CompareSignatures(spark)
    for x, y in candidate_pairs:
        result = compare_signatures.compare_signatures_spark(
            signatures[x], signatures[y]
        )
        logger.info(f"For couple {x},{y} similarity rate = {result:.2f}")

    complete_time_end = time.time()
    logger.info(
        f"Time elapsed for Comparing signatures: {complete_time_end - compare_sign_start_time:.3f} seconds"
    )

    elapsed_total_time = complete_time_end - complete_time_start
    logger.info(
        f"Total time elapsed for the whole procedure: {elapsed_total_time:.3f} seconds"
    )

def test_books_no_spark(rows_per_band: int = 4):
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

    complete_time_start = time.time()

    shingling = Shingling(spark)

    shingles = shingling.shingle_multi(documents)

    # Need to maintain a "per document" logic
    hash_multi = shingling.hash_multi(shingles)

    shingling_time = time.time()
    logger.info(
        f"Time elapsed for shingling (no spark): {shingling_time - complete_time_start:.3f} seconds"
    )

    minhashing_start_time = time.time()

    minhashing = Minhashing(spark, n=100)
    signatures = [minhashing.get_signature(h) for h in hash_multi]

    minhashing_end_time = time.time()
    logger.info(
        f"Time elapsed for minhashing (no spark): {minhashing_end_time - minhashing_start_time:.3f} seconds"
    )

    lsh_start_time = time.time()

    lsh = LSH(spark, n=100, r=rows_per_band)
    candidate_pairs = lsh.get_candidate_pairs(signatures)
    logger.info(f"Candidate pairs for r={rows_per_band}: {candidate_pairs}")

    lsh_end_time = time.time()
    logger.info(f"Time elapsed for LSH (no spark): {lsh_end_time - lsh_start_time:.3f} seconds")

    print(f"{candidate_pairs}")

    compare_sign_start_time = time.time()

    compare_signatures = CompareSignatures(spark)
    for x, y in candidate_pairs:
        result = compare_signatures.compare_signatures(
            signatures[x], signatures[y]
        )
        logger.info(f"For couple {x},{y} similarity rate = {result:.2f}")

    complete_time_end = time.time()
    logger.info(
        f"Time elapsed for Comparing signatures (no spark): {complete_time_end - compare_sign_start_time:.3f} seconds"
    )

    elapsed_total_time = complete_time_end - complete_time_start
    logger.info(
        f"Total time elapsed for the whole procedure (no spark): {elapsed_total_time:.3f} seconds"
    )


def test_books_allan_poe(rows_per_band: int = 3):
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
        if author == "Edgar Allan Poe":
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

    complete_time_start = time.time()

    shingling = Shingling(spark)

    shingles = shingling.shingle_multi(documents)

    # Need to maintain a "per document" logic
    hash_multi = shingling.hash_multi_spark(shingles)

    shingling_time = time.time()
    logger.info(
        f"Time elapsed for shingling: {shingling_time - complete_time_start:.3f} seconds"
    )

    minhashing_start_time = time.time()

    minhashing = Minhashing(spark, n=100)
    signatures = [minhashing.get_signature(h) for h in hash_multi]

    minhashing_end_time = time.time()
    logger.info(
        f"Time elapsed for minhashing: {minhashing_end_time - minhashing_start_time:.3f} seconds"
    )

    lsh_start_time = time.time()

    lsh = LSH(spark, n=100, r=rows_per_band)
    candidate_pairs = lsh.get_candidate_pairs_spark(signatures)
    logger.info(f"Candidate pairs for r={rows_per_band}: {candidate_pairs}")

    lsh_end_time = time.time()
    logger.info(f"Time elapsed for LSH: {lsh_end_time - lsh_start_time:.3f} seconds")

    print(f"{candidate_pairs}")

    compare_sign_start_time = time.time()

    compare_signatures = CompareSignatures(spark)
    for x, y in candidate_pairs:
        result = compare_signatures.compare_signatures_spark(
            signatures[x], signatures[y]
        )
        logger.info(f"For couple {x},{y} similarity rate = {result:.2f}")

    complete_time_end = time.time()
    logger.info(
        f"Time elapsed for Comparing signatures: {complete_time_end - compare_sign_start_time:.3f} seconds"
    )

    elapsed_total_time = complete_time_end - complete_time_start
    logger.info(
        f"Total time elapsed for the whole procedure: {elapsed_total_time:.3f} seconds"
    )


if __name__ == "__main__":
    test_books()
    test_books_no_spark()
    # test_books_allan_poe()
