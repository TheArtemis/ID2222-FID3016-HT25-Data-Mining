import logging
from pathlib import Path
import time

from miner.core.triest.triest import TriestBase

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "triest" / "facebook_combined.txt"
logger = logging.getLogger(__name__)

# size of the memory; it could be modified
M = 10000


def test_base(M):
    start_time = time.time()
    triest = TriestBase(M)
    estimations = []
    n = 5
    for i in range(0, n):
        logger.info(f"Running iteration {i + 1} for M={M}...")
        estimation = triest.run(DATASET_PATH)
        if callable(estimation):
            estimation_value = estimation()
        else:
            estimation_value = estimation
        estimations.append(estimation_value)
    end_time = time.time()

    for estimation in estimations:
        logger.info(f"{estimation}")

    time_needed = (end_time - start_time) / float(n)
    logger.info(
        f"Average time needed for the experiment is: {time_needed:.2f} seconds "
    )


if __name__ == "__main__":
    test_base(M)
