import logging
from pathlib import Path
import time

from miner.core.triest.triest import TriestBase

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "triest" / "facebook_combined.txt"
logger = logging.getLogger(__name__)

#size of the memory; it could be modified
M = 1000

def test_base(M):
    triest = TriestBase(M)
    s = triest.run(DATASET_PATH)

    logger.info(f's = {s}')


if __name__ == "__main__":
    test_base(M)
