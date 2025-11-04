from pathlib import Path
import csv
import sys


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

    for line in dickens_matrix:
        print(f"{line}")


if __name__ == "__main__":
    test_books()
