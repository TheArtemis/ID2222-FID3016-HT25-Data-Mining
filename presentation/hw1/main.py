from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import the test module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_test(name: str = "books") -> None:
    """Run one of the small demo tests.

    The function imports the test module lazily to avoid starting Spark
    during module import.
    """
    from test.similarity import test_books as tb  # local import

    mapping = {
        "dickens": tb.test_books,
        "dickens_no_spark": tb.test_books_no_spark,
        "allan_poe": tb.test_books_allan_poe,
    }

    try:
        func = mapping[name]
    except KeyError:
        raise SystemExit(f"Unknown test: {name}. Choose from: {', '.join(mapping)}")

    func()


if __name__ == "__main__":
    import argparse

    choices = ["dickens", "dickens_no_spark", "allan_poe"]
    parser = argparse.ArgumentParser(
        prog="hw1-run", description="Run a demo test from hw1"
    )
    parser.add_argument(
        "test", nargs="?", default=None, choices=choices, help="which test to run"
    )
    args = parser.parse_args()

    if args.test is None:
        prompt = f"Select test ({', '.join(choices)}) [default: allan_poe]: "
        while True:
            try:
                ans = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()  # newline on Ctrl-D/Ctrl-C
                args.test = "allan_poe"
                break
            if ans == "":
                args.test = "allan_poe"
                break
            if ans in choices:
                args.test = ans
                break
            print(f"Invalid choice: {ans}. Choose one of: {', '.join(choices)}")
    run_test(args.test)
