import logging
from pathlib import Path
import time

from miner.core.triest.triest import TriestBase

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "triest" / "facebook_combined.txt"
logger = logging.getLogger(__name__)

#size of the memory; it could be modified
M = 50000

def test_base(M):
    triest = TriestBase(M)
    estimations = []
    for i in range(0,1):
        logger.info(f"Running iteration {i+1} for M={M}...")
        estimation = triest.run(DATASET_PATH)
        estimations.append(estimation)
    
    for estimation in estimations:
        logger.info(f'{estimation}')


if __name__ == "__main__":
    test_base(M)
