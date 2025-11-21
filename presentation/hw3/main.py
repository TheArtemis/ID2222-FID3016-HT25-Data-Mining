from pathlib import Path
import sys
import logging

# Ensure project root is on sys.path so we can import the test modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing test modules for Triest
from test.triest import test_base as tb
from test.triest import test_improved as ti


def run_base(M: int) -> None:
    """Delegate to test/triest/test_base.py:test_base with configurable M"""
    tb.test_base(M)


def run_improved(M: int) -> None:
    """Delegate to test/triest/test_improved.py:test_improved with configurable M"""
    ti.test_improved(M)


def run_selected(name: str) -> None:
    if name == "base":
        run_base()
    elif name == "improved":
        run_improved()
    else:
        raise SystemExit(f"Unknown test: {name}. Choose from: base, improved")


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="hw3-run", description="Run hw3 Triest demos")
    parser.add_argument("test", nargs="?", choices=["base", "improved"], default=None)
    parser.add_argument("--M", type=int, default=None, help="Memory size M to use for Triest (overrides module default)")
    args = parser.parse_args()

    if args.test is None:
        prompt = "Select test (base, improved) [default: base]: "
        while True:
            try:
                ans = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                ans = "base"
            if ans == "":
                ans = "base"
            if ans in ("base", "improved"):
                args.test = ans
                break
            print(f"Invalid choice: {ans}. Choose one of: base, improved")

    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # determine M (use passed value or module default)
    if args.M is not None:
        M_val = args.M
    else:
        M_val = tb.M if args.test in (None, "base") else ti.M

    # check that the dataset file exists before running
    dataset_path = Path(tb.DATASET_PATH)
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}\nPlease ensure the file exists or update the DATASET_PATH in the test modules.")
        raise SystemExit(1)

    # run the selected test with M_val
    if args.test == "improved":
        run_improved(M_val)
    else:
        run_base(M_val)


if __name__ == "__main__":
    main()
